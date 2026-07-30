import {
  createBashToolDefinition,
  createFindToolDefinition,
  createGrepToolDefinition,
  createLocalBashOperations,
  createLsToolDefinition,
  createReadToolDefinition
} from "@earendil-works/pi-coding-agent";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { execFile } from "node:child_process";
import { constants, existsSync } from "node:fs";
import {
  access,
  chmod,
  copyFile,
  mkdir,
  readFile,
  readdir,
  realpath,
  stat,
  writeFile
} from "node:fs/promises";
import { homedir } from "node:os";
import {
  delimiter,
  dirname,
  isAbsolute,
  join,
  matchesGlob,
  relative,
  resolve,
  sep
} from "node:path";
import { promisify } from "node:util";

export type ExecutorSandboxMode = "macos-seatbelt" | "linux-bubblewrap" | "linux-docker" | "workspace";

// Default per-call bash timeout (seconds) when the model does not pass one.
// The Pi SDK schema leaves timeout optional with no default; without a floor a
// runaway command (e.g. an unbounded brute-force loop) stalls the epoch until
// the global run deadline. The SDK kills the whole process tree on timeout and
// returns a "Command timed out" tool error the Executor can react to.
function executorBashDefaultTimeoutSeconds(): number {
  const value = Number(process.env.EXECUTOR_BASH_DEFAULT_TIMEOUT_S);
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 300;
}

export type ExecutorSandboxRequestedMode = "auto" | "seatbelt" | "bubblewrap" | "docker" | "workspace";

export type ExecutorSandbox = {
  root: string;
  mode: ExecutorSandboxMode;
  profilePath?: string;
  backendPath?: string;
  networkMode?: ExecutorDockerNetworkMode;
  allowedReadRoots: string[];
  createTools: () => ToolDefinition<any, any, any>[];
};

export type ExecutorDockerNetworkMode = "host" | "bridge" | "none";

const execFileAsync = promisify(execFile);
const SANDBOX_BACKEND_STARTUP_TIMEOUT_MS = 10_000;

export async function createExecutorSandbox(input: {
  runtimeDir: string;
  runId: string;
  mode?: ExecutorSandboxRequestedMode;
  environment?: NodeJS.ProcessEnv;
  additionalReadRoots?: string[];
}): Promise<ExecutorSandbox> {
  const runtimeDir = resolve(input.runtimeDir);
  const root = join(runtimeDir, "sandboxes", input.runId);
  const home = join(root, "home");
  const temp = join(root, "tmp");
  await Promise.all([mkdir(root, { recursive: true }), mkdir(home, { recursive: true }), mkdir(temp, { recursive: true })]);
  const canonicalRoot = await realpath(root);
  const environment = await prepareSandboxEnvironment(input.environment, canonicalRoot);
  const allowedReadRoots = await existingCanonicalRoots([
    canonicalRoot,
    join(homedir(), ".agents", "skills"),
    join(homedir(), ".codex", "skills"),
    join(homedir(), ".pi", "agent", "skills"),
    ...(input.additionalReadRoots ?? [])
  ]);
  const requestedMode = input.mode ?? executorSandboxModeFromEnv();
  const seatbeltPath = process.platform === "darwin" && existsSync("/usr/bin/sandbox-exec")
    ? "/usr/bin/sandbox-exec"
    : undefined;
  const bubblewrapPath = process.platform === "linux" || requestedMode === "bubblewrap"
    ? await findExecutable(process.env.BWRAP_PATH, "bwrap")
    : undefined;
  const dockerPath = process.platform === "linux"
    ? await findExecutable(process.env.DOCKER_PATH, "docker")
    : undefined;
  const dockerImage = process.env.EXECUTOR_SANDBOX_DOCKER_IMAGE?.trim();
  if (requestedMode === "seatbelt" && !seatbeltPath) {
    throw new Error("EXECUTOR_SANDBOX_MODE=seatbelt requires macOS sandbox-exec");
  }
  if (requestedMode === "bubblewrap" && !bubblewrapPath) {
    throw new Error("EXECUTOR_SANDBOX_MODE=bubblewrap requires the bwrap executable");
  }
  if (requestedMode === "docker" && process.platform !== "linux") {
    throw new Error("EXECUTOR_SANDBOX_MODE=docker is supported only on Linux");
  }
  if (requestedMode === "docker" && !dockerPath) {
    throw new Error("EXECUTOR_SANDBOX_MODE=docker requires the docker executable");
  }
  if (requestedMode === "docker" && !dockerImage) {
    throw new Error("EXECUTOR_SANDBOX_MODE=docker requires EXECUTOR_SANDBOX_DOCKER_IMAGE");
  }
  const useSeatbelt = requestedMode === "seatbelt"
    || (requestedMode === "auto" && Boolean(seatbeltPath));
  let useBubblewrap = requestedMode === "bubblewrap"
    || (requestedMode === "auto" && !useSeatbelt && Boolean(bubblewrapPath));
  let useDocker = requestedMode === "docker"
    || (requestedMode === "auto" && !useSeatbelt && !useBubblewrap && Boolean(dockerPath && dockerImage));
  const readOnlyRoots = allowedReadRoots.filter((candidate) => candidate !== canonicalRoot);
  if (useBubblewrap && bubblewrapPath) {
    try {
      await verifyBubblewrapUsable({
        bubblewrapPath,
        sandboxRoot: canonicalRoot,
        readOnlyRoots
      });
    } catch (error) {
      if (requestedMode === "auto" && dockerPath && dockerImage) {
        useBubblewrap = false;
        useDocker = true;
      } else {
        throw error;
      }
    }
  }
  if (requestedMode === "auto" && process.platform === "linux" && !useBubblewrap && !useDocker) {
    throw new Error(
      "Executor sandbox auto mode requires a usable Bubblewrap backend or a configured Docker backend; set EXECUTOR_SANDBOX_MODE=workspace explicitly to run without either backend"
    );
  }
  const dockerNetworkMode = useDocker ? executorDockerNetworkModeFromEnv() : undefined;
  const dockerUser = useDocker ? currentProcessUser() : undefined;
  if (useDocker && !dockerUser) {
    throw new Error("EXECUTOR_SANDBOX_MODE=docker requires a resolvable process uid and gid");
  }
  if (useDocker && dockerPath && dockerImage) {
    await verifyDockerImageAvailable(dockerPath, dockerImage);
  }
  const mode: ExecutorSandboxMode = useSeatbelt
    ? "macos-seatbelt"
    : useBubblewrap
      ? "linux-bubblewrap"
      : useDocker
        ? "linux-docker"
        : "workspace";
  const profilePath = useSeatbelt ? join(runtimeDir, `executor-${input.runId}.sb`) : undefined;
  if (profilePath) {
    await writeFile(profilePath, createSeatbeltProfile({
      sandboxRoot: canonicalRoot,
      readOnlyRoots: allowedReadRoots.filter((candidate) => candidate !== canonicalRoot)
    }), "utf8");
  }
  const pathPolicy = new SandboxPathPolicy(canonicalRoot, allowedReadRoots);
  const localBash = createLocalBashOperations();

  return {
    root: canonicalRoot,
    mode,
    profilePath,
    backendPath: useSeatbelt ? seatbeltPath : useBubblewrap ? bubblewrapPath : useDocker ? dockerPath : undefined,
    networkMode: dockerNetworkMode,
    allowedReadRoots,
    createTools: () => [
      createReadToolDefinition(canonicalRoot, {
        operations: {
          access: async (absolutePath) => {
            const readablePath = await pathPolicy.requireReadable(absolutePath);
            await access(readablePath, constants.R_OK);
          },
          readFile: async (absolutePath) => readFile(await pathPolicy.requireReadable(absolutePath))
        }
      }),
      createBashToolDefinition(canonicalRoot, {
        operations: {
          exec: async (command, _cwd, options) => {
            const commandEnvironment = sandboxEnvironment(
              mergeCommandEnvironment(options.env, environment),
              canonicalRoot
            );
            const dockerContainerName = dockerPath && dockerImage && useDocker
              ? createDockerContainerName(input.runId)
              : undefined;
            const wrappedCommand = profilePath
              ? `${shellQuote(seatbeltPath!)} -f ${shellQuote(profilePath)} /bin/zsh --emulate sh -f -c ${shellQuote(command)}`
              : bubblewrapPath && useBubblewrap
                ? renderShellCommand(createBubblewrapCommand({
                    bubblewrapPath,
                    sandboxRoot: canonicalRoot,
                    readOnlyRoots,
                    command
                  }))
                : dockerPath && dockerImage && useDocker && dockerContainerName && dockerNetworkMode && dockerUser
                  ? renderShellCommand(createDockerSandboxCommand({
                      dockerPath,
                      image: dockerImage,
                      sandboxRoot: canonicalRoot,
                      readOnlyRoots,
                      command,
                      containerName: dockerContainerName,
                      networkMode: dockerNetworkMode,
                      user: dockerUser,
                      environment: commandEnvironment
                    }))
                  : command;
            try {
              return await localBash.exec(wrappedCommand, canonicalRoot, {
                ...options,
                timeout: options.timeout ?? executorBashDefaultTimeoutSeconds(),
                env: commandEnvironment
              });
            } finally {
              if (dockerPath && dockerContainerName && useDocker) {
                await removeDockerContainer(dockerPath, dockerContainerName);
              }
            }
          }
        }
      }),
      createGrepToolDefinition(canonicalRoot, {
        operations: {
          isDirectory: async (absolutePath) => (await stat(await pathPolicy.requireReadable(absolutePath))).isDirectory(),
          readFile: async (absolutePath) => readFile(await pathPolicy.requireReadable(absolutePath), "utf8")
        }
      }),
      createFindToolDefinition(canonicalRoot, {
        operations: {
          exists: async (absolutePath) => {
            await pathPolicy.requireReadable(absolutePath);
            return true;
          },
          glob: async (pattern, searchRoot, options) => findWithinRoot({
            pattern,
            searchRoot: await pathPolicy.requireReadable(searchRoot),
            ignore: options.ignore,
            limit: options.limit
          })
        }
      }),
      createLsToolDefinition(canonicalRoot, {
        operations: {
          exists: async (absolutePath) => {
            await pathPolicy.requireReadable(absolutePath);
            return true;
          },
          stat: async (absolutePath) => stat(await pathPolicy.requireReadable(absolutePath)),
          readdir: async (absolutePath) => readdir(await pathPolicy.requireReadable(absolutePath))
        }
      })
    ] as ToolDefinition<any, any, any>[]
  };
}

export class SandboxPathPolicy {
  constructor(
    readonly root: string,
    readonly allowedReadRoots: string[] = [root]
  ) {}

  async requireReadable(candidatePath: string): Promise<string> {
    const resolvedPath = resolve(this.root, candidatePath);
    let canonicalPath: string;
    try {
      canonicalPath = await realpath(resolvedPath);
    } catch {
      throw new Error(`Executor sandbox path does not exist: ${candidatePath}`);
    }
    if (!this.allowedReadRoots.some((allowedRoot) => isWithin(allowedRoot, canonicalPath))) {
      throw new Error(`Executor sandbox denied path outside allowed roots: ${candidatePath}`);
    }
    return canonicalPath;
  }
}

export function createSeatbeltProfile(input: {
  sandboxRoot: string;
  readOnlyRoots?: string[];
}): string {
  const readableRoots = [
    input.sandboxRoot,
    ...(input.readOnlyRoots ?? []),
    "/opt",
    "/usr/local",
    "/private/etc",
    "/private/var/select"
  ];
  return [
    "(version 1)",
    "(import \"system.sb\")",
    "(allow process*)",
    "(allow network*)",
    `(allow file-read-metadata (subpath ${seatbeltString(homedir())}) (subpath \"/private/var/folders\"))`,
    `(allow file-read* ${readableRoots.map((root) => `(subpath ${seatbeltString(root)})`).join(" ")})`,
    `(allow file-write* (subpath ${seatbeltString(input.sandboxRoot)}))`
  ].join("\n") + "\n";
}

export function createBubblewrapCommand(input: {
  bubblewrapPath: string;
  sandboxRoot: string;
  readOnlyRoots?: string[];
  command: string;
  shellPath?: string;
}): string[] {
  const shellPath = input.shellPath ?? firstExistingPath(["/bin/bash", "/usr/bin/bash", "/bin/sh"]) ?? "/bin/sh";
  const systemRoots = existingPaths(["/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt"]);
  const systemFiles = existingPaths([
    "/etc/hosts",
    "/etc/resolv.conf",
    "/etc/nsswitch.conf",
    "/etc/services",
    "/etc/protocols",
    "/etc/passwd",
    "/etc/group",
    "/etc/localtime",
    "/etc/ld.so.cache"
  ]);
  const systemDirectories = existingPaths([
    "/etc/alternatives",
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/pki",
    "/etc/ld.so.conf.d",
    "/run/systemd/resolve",
    "/run/NetworkManager"
  ]);
  const argumentsList = [
    input.bubblewrapPath,
    "--die-with-parent",
    "--new-session",
    "--unshare-pid",
    "--unshare-ipc",
    "--unshare-uts",
    "--proc",
    "/proc",
    "--dev",
    "/dev",
    "--dir",
    "/tmp",
    "--dir",
    "/etc",
    "--dir",
    "/home",
    "--dir",
    "/run"
  ];
  for (const root of dedupeNestedRoots([...systemRoots, ...systemDirectories, ...(input.readOnlyRoots ?? [])])) {
    argumentsList.push("--ro-bind", root, root);
  }
  for (const file of systemFiles) {
    argumentsList.push("--ro-bind", file, file);
  }
  argumentsList.push(
    "--bind",
    input.sandboxRoot,
    input.sandboxRoot,
    "--bind",
    join(input.sandboxRoot, "tmp"),
    "/tmp",
    "--chdir",
    input.sandboxRoot,
    "--setenv",
    "HOME",
    join(input.sandboxRoot, "home"),
    "--setenv",
    "TMPDIR",
    join(input.sandboxRoot, "tmp"),
    "--setenv",
    "TMPPREFIX",
    join(input.sandboxRoot, "tmp", "zsh"),
    shellPath,
    "-c",
    input.command
  );
  return argumentsList;
}

async function verifyBubblewrapUsable(input: {
  bubblewrapPath: string;
  sandboxRoot: string;
  readOnlyRoots: string[];
}): Promise<void> {
  const [commandPath, ...argumentsList] = createBubblewrapCommand({
    bubblewrapPath: input.bubblewrapPath,
    sandboxRoot: input.sandboxRoot,
    readOnlyRoots: input.readOnlyRoots,
    command: "true"
  });
  try {
    await execFileAsync(commandPath, argumentsList, {
      timeout: SANDBOX_BACKEND_STARTUP_TIMEOUT_MS,
      maxBuffer: 8 * 1024,
      windowsHide: true
    });
  } catch (error) {
    throw new Error(
      `Bubblewrap startup check failed: Linux namespace setup is unavailable (${commandFailureDiagnostic(error)})`
    );
  }
}

/**
 * Run commands in a non-privileged container when the host cannot create a
 * user namespace for Bubblewrap. The per-run workspace is writable and
 * explicitly allowed skill roots are mounted read-only. The Docker socket and
 * all other host paths remain outside the container.
 */
export function createDockerSandboxCommand(input: {
  dockerPath: string;
  image: string;
  sandboxRoot: string;
  readOnlyRoots?: string[];
  command: string;
  containerName: string;
  networkMode: ExecutorDockerNetworkMode;
  shellPath?: string;
  user: string;
  environment?: NodeJS.ProcessEnv;
}): string[] {
  const shellPath = input.shellPath ?? "/bin/bash";
  const home = join(input.sandboxRoot, "home");
  const argumentsList = [
    input.dockerPath,
    "run",
    "--rm",
    "--init",
    "--pull",
    "never",
    "--name",
    input.containerName,
    "--network",
    input.networkMode,
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "256",
    "--memory",
    "1g",
    "--tmpfs",
    "/run:rw,nosuid,nodev,noexec,size=16m",
    "--mount",
    `type=bind,source=${input.sandboxRoot},target=${input.sandboxRoot},bind-propagation=rprivate`,
    "--mount",
    `type=bind,source=${join(input.sandboxRoot, "tmp")},target=/tmp,bind-propagation=rprivate`,
    "--workdir",
    input.sandboxRoot,
    "--env",
    `HOME=${home}`,
    "--env",
    "TMPDIR=/tmp",
    "--env",
    "TMPPREFIX=/tmp/zsh"
  ];
  for (const root of dedupeNestedRoots(input.readOnlyRoots ?? [])) {
    argumentsList.push(
      "--mount",
      `type=bind,source=${root},target=${root},readonly,bind-propagation=rprivate`
    );
  }
  for (const name of dockerEnvironmentNames(input.environment)) {
    argumentsList.push("--env", name);
  }
  argumentsList.push("--user", input.user);
  argumentsList.push(
    "--entrypoint",
    shellPath,
    input.image,
    "-c",
    input.command
  );
  return argumentsList;
}

async function verifyDockerImageAvailable(dockerPath: string, image: string): Promise<void> {
  try {
    await execFileAsync(dockerPath, ["image", "inspect", "--format", "{{.Id}}", image], {
      timeout: SANDBOX_BACKEND_STARTUP_TIMEOUT_MS,
      maxBuffer: 8 * 1024,
      windowsHide: true
    });
  } catch (error) {
    throw new Error(
      `Docker sandbox image is unavailable (${commandFailureDiagnostic(error)}); pre-load ${image} before starting the run`
    );
  }
}

async function removeDockerContainer(dockerPath: string, containerName: string): Promise<void> {
  try {
    await execFileAsync(dockerPath, ["rm", "--force", containerName], {
      timeout: SANDBOX_BACKEND_STARTUP_TIMEOUT_MS,
      maxBuffer: 8 * 1024,
      windowsHide: true
    });
  } catch {
    // The normal --rm path has already removed the container. Cleanup must
    // never replace a tool result with a harmless "not found" response.
  }
}

function createDockerContainerName(runId: string): string {
  const normalizedRunId = runId.toLowerCase().replaceAll(/[^a-z0-9_.-]/g, "-").slice(0, 36) || "run";
  const suffix = `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  return `luan1ao-${normalizedRunId}-${suffix}`.slice(0, 120);
}

function dockerEnvironmentNames(environment: NodeJS.ProcessEnv | undefined): string[] {
  if (!environment) return [];
  const allowedNames = [
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "PYTHONDONTWRITEBYTECODE"
  ];
  return allowedNames.filter((name) => typeof environment[name] === "string");
}

function currentProcessUser(): string | undefined {
  if (typeof process.getuid !== "function" || typeof process.getgid !== "function") {
    return undefined;
  }
  return `${process.getuid()}:${process.getgid()}`;
}

function executorDockerNetworkModeFromEnv(): ExecutorDockerNetworkMode {
  const value = process.env.EXECUTOR_SANDBOX_DOCKER_NETWORK?.trim().toLowerCase() ?? "host";
  if (value === "host" || value === "bridge" || value === "none") {
    return value;
  }
  throw new Error("EXECUTOR_SANDBOX_DOCKER_NETWORK must be host, bridge, or none");
}

function executorSandboxModeFromEnv(): ExecutorSandboxRequestedMode {
  const value = process.env.EXECUTOR_SANDBOX_MODE?.trim().toLowerCase();
  if (value === "bwrap" || value === "linux-bubblewrap") {
    return "bubblewrap";
  }
  if (value === "linux-docker") {
    return "docker";
  }
  if (value === "seatbelt" || value === "bubblewrap" || value === "docker" || value === "workspace") {
    return value;
  }
  return "auto";
}

async function prepareSandboxEnvironment(input: NodeJS.ProcessEnv | undefined, root: string): Promise<NodeJS.ProcessEnv | undefined> {
  if (!input?.HTTP_PROXY && !input?.http_proxy) return input;
  const sourceCaPath = input.CURL_CA_BUNDLE ?? input.SSL_CERT_FILE ?? input.NODE_EXTRA_CA_CERTS;
  if (!sourceCaPath) throw new Error("managed proxy environment has no public CA certificate");
  const sandboxCaPath = join(root, "traffic-proxy-ca.crt");
  await copyFile(sourceCaPath, sandboxCaPath);
  await chmod(sandboxCaPath, 0o444);
  return {
    ...input,
    SSL_CERT_FILE: sandboxCaPath,
    CURL_CA_BUNDLE: sandboxCaPath,
    NODE_EXTRA_CA_CERTS: sandboxCaPath
  };
}

function mergeCommandEnvironment(
  toolEnvironment: NodeJS.ProcessEnv | undefined,
  managedEnvironment: NodeJS.ProcessEnv | undefined
): NodeJS.ProcessEnv | undefined {
  if (!managedEnvironment) return toolEnvironment;
  const environment = { ...toolEnvironment, ...managedEnvironment };
  if (managedEnvironment.HTTP_PROXY || managedEnvironment.http_proxy) {
    delete environment.ALL_PROXY;
    delete environment.all_proxy;
    delete environment.NO_PROXY;
    delete environment.no_proxy;
  }
  return environment;
}

function sandboxEnvironment(input: NodeJS.ProcessEnv | undefined, root: string): NodeJS.ProcessEnv {
  const source = input ?? process.env;
  const output: NodeJS.ProcessEnv = {
    HOME: join(root, "home"),
    TMPDIR: join(root, "tmp"),
    TMPPREFIX: join(root, "tmp", "zsh"),
    PATH: source.PATH ?? "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    LANG: source.LANG ?? "C.UTF-8",
    LC_ALL: source.LC_ALL,
    LC_CTYPE: source.LC_CTYPE,
    TERM: source.TERM,
    TZ: source.TZ,
    HTTP_PROXY: source.HTTP_PROXY,
    HTTPS_PROXY: source.HTTPS_PROXY,
    http_proxy: source.http_proxy,
    https_proxy: source.https_proxy,
    NO_PROXY: source.NO_PROXY,
    no_proxy: source.no_proxy,
    SSL_CERT_FILE: source.SSL_CERT_FILE,
    SSL_CERT_DIR: source.SSL_CERT_DIR,
    CURL_CA_BUNDLE: source.CURL_CA_BUNDLE,
    NODE_EXTRA_CA_CERTS: source.NODE_EXTRA_CA_CERTS,
    PYTHONDONTWRITEBYTECODE: "1"
  };
  return Object.fromEntries(Object.entries(output).filter((entry): entry is [string, string] => typeof entry[1] === "string"));
}

async function existingCanonicalRoots(candidates: string[]): Promise<string[]> {
  const roots: string[] = [];
  for (const candidate of candidates) {
    try {
      roots.push(await realpath(candidate));
    } catch {
      // Optional read-only roots are omitted when unavailable.
    }
  }
  return [...new Set(roots)];
}

async function findExecutable(explicitPath: string | undefined, executableName: string): Promise<string | undefined> {
  const candidates = explicitPath
    ? [explicitPath]
    : (process.env.PATH ?? "").split(delimiter).filter(Boolean).map((pathEntry) => join(pathEntry, executableName));
  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK);
      return await realpath(candidate);
    } catch {
      continue;
    }
  }
  return undefined;
}

function existingPaths(paths: string[]): string[] {
  return paths.filter((candidate) => existsSync(candidate));
}

function firstExistingPath(paths: string[]): string | undefined {
  return paths.find((candidate) => existsSync(candidate));
}

function dedupeNestedRoots(paths: string[]): string[] {
  const uniquePaths = [...new Set(paths)].sort((left, right) => left.length - right.length);
  return uniquePaths.filter((candidate, index) => !uniquePaths.slice(0, index).some((parent) => isWithin(parent, candidate)));
}

function renderShellCommand(argumentsList: string[]): string {
  return argumentsList.map(shellQuote).join(" ");
}

async function findWithinRoot(input: {
  pattern: string;
  searchRoot: string;
  ignore: string[];
  limit: number;
}): Promise<string[]> {
  const results: string[] = [];
  const ignoredNames = new Set(["node_modules", ".git"]);
  const visit = async (currentDirectory: string): Promise<void> => {
    if (results.length >= input.limit) {
      return;
    }
    const entries = await readdir(currentDirectory, { withFileTypes: true });
    for (const entry of entries) {
      if (results.length >= input.limit) {
        return;
      }
      if (ignoredNames.has(entry.name) || entry.isSymbolicLink()) {
        continue;
      }
      const absolutePath = join(currentDirectory, entry.name);
      const relativePath = relative(input.searchRoot, absolutePath).split(sep).join("/");
      if (entry.isDirectory()) {
        await visit(absolutePath);
        continue;
      }
      if (matchesGlob(relativePath, input.pattern) || (!input.pattern.includes("/") && matchesGlob(entry.name, input.pattern))) {
        results.push(absolutePath);
      }
    }
  };
  await visit(input.searchRoot);
  return results;
}

function isWithin(root: string, candidate: string): boolean {
  const relativePath = relative(root, candidate);
  return relativePath === "" || (!relativePath.startsWith(`..${sep}`) && relativePath !== ".." && !isAbsolute(relativePath));
}

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`;
}

function commandFailureDiagnostic(error: unknown): string {
  const candidate = error as { stderr?: unknown; message?: unknown };
  const detail = typeof candidate.stderr === "string" && candidate.stderr.trim()
    ? candidate.stderr
    : typeof candidate.message === "string"
      ? candidate.message
      : "unknown backend error";
  return detail.replaceAll(/\s+/g, " ").trim().slice(0, 500);
}

function seatbeltString(value: string): string {
  return JSON.stringify(value);
}
