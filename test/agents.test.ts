import assert from "node:assert/strict";
import test from "node:test";
import { Check } from "typebox/value";
import { createExecutorResearchTools, observerToolsForMode } from "../src/agents.js";
import { createGraphDeltaSubmitTool, createPlannerSubmitTool } from "../src/tools/pi-tools.js";
import type { ArtifactStore } from "../src/stores/artifact-store.js";
import type { ExecutionLog } from "../src/stores/execution-log.js";
import type { SQLiteGraphStore } from "../src/stores/graph-store.js";

test("supervisor observer mode exposes only the terminating control tool", () => {
  const tools = observerToolsForMode({
    mode: "supervise",
    graphStore: {} as SQLiteGraphStore,
    executionLog: {} as ExecutionLog,
    artifactStore: {} as ArtifactStore
  });

  assert.deepEqual(tools.map((tool) => tool.name), ["control_submit"]);
});

test("projector observer mode exposes bounded read-only graph tools and the terminating graph tool", () => {
  const tools = observerToolsForMode({
    mode: "project",
    graphStore: {} as SQLiteGraphStore,
    executionLog: {} as ExecutionLog,
    artifactStore: {} as ArtifactStore
  });

  assert.deepEqual(tools.map((tool) => tool.name), [
    "graph_search",
    "graph_query",
    "graph_trace",
    "graph_delta_submit"
  ]);
});

test("executor exposes bounded public research tools", () => {
  assert.deepEqual(
    createExecutorResearchTools().map((tool) => tool.name),
    ["web_fetch", "web_search", "vulnerability_search"]
  );
});

test("projector terminal tool accepts bounded task-shaped drafts for deterministic sanitization", () => {
  const tool = createGraphDeltaSubmitTool();
  const schema = tool.parameters as unknown as {
    properties: {
      nodes: {
        maxItems?: number;
        items?: {
          anyOf?: Array<{
            additionalProperties?: boolean;
            properties?: {
              id?: { pattern?: string };
              graphKind?: { const?: string };
              properties?: { additionalProperties?: { anyOf?: unknown[] } };
            };
          }>;
        };
      };
      edges: {
        maxItems?: number;
        items?: {
          additionalProperties?: boolean;
          properties?: {
            from?: { pattern?: string };
            to?: { pattern?: string };
            type?: { anyOf?: Array<{ const?: string }> };
          };
        };
      };
      sourceEventIds?: unknown;
    };
    additionalProperties?: boolean;
  };

  assert.equal(schema.properties.nodes.maxItems, 12);
  assert.equal(schema.properties.edges.maxItems, 20);
  assert.deepEqual(
    schema.properties.nodes.items?.anyOf?.map((branch) => branch.properties?.graphKind?.const),
    ["reasoning", "operation", "task"]
  );
  assert.ok(schema.properties.nodes.items?.anyOf?.every((branch) => branch.additionalProperties === false));
  assert.ok(schema.properties.nodes.items?.anyOf?.every((branch) => branch.properties?.id?.pattern === "^(existing|new):[1-9][0-9]*$"));
  assert.ok(schema.properties.nodes.items?.anyOf?.every((branch) => (branch.properties?.properties?.additionalProperties?.anyOf?.length ?? 0) > 0));
  assert.equal(schema.properties.edges.items?.additionalProperties, false);
  assert.equal(schema.properties.edges.items?.properties?.from?.pattern, "^(existing|new):[1-9][0-9]*$");
  assert.equal(schema.properties.edges.items?.properties?.to?.pattern, "^(existing|new):[1-9][0-9]*$");
  assert.equal(
    schema.properties.edges.items?.properties?.type?.anyOf?.some((branch) => branch.const === "depends_on"),
    true
  );
  assert.equal(schema.properties.sourceEventIds, undefined);
  assert.equal(schema.additionalProperties, false);
});

test("projector terminal tool prepares JSON-serialized node and edge arrays before strict validation", () => {
  const tool = createGraphDeltaSubmitTool();
  const nodes = [{
    id: "new:1",
    label: "HTTP service",
    graphKind: "operation",
    type: "Service",
    properties: { product: "example" },
    evidenceRefs: ["o1"]
  }];
  const edges: [] = [];
  const prepareArguments = tool.prepareArguments;
  assert.ok(prepareArguments);
  const prepared = prepareArguments({ nodes: JSON.stringify(nodes), edges: JSON.stringify(edges) });

  assert.deepEqual(prepared, { nodes, edges });
  assert.equal(Check(tool.parameters, prepared), true);
  assert.deepEqual(prepareArguments({ nodes, edges }), { nodes, edges });
});

test("projector terminal tool leaves invalid serialized fields for strict schema rejection", () => {
  const tool = createGraphDeltaSubmitTool();
  const prepareArguments = tool.prepareArguments;
  assert.ok(prepareArguments);
  const malformed = { nodes: "{not-json", edges: "[]" };
  const nonArray = { nodes: "{}", edges: "[]" };
  const nestedString = { nodes: JSON.stringify(["not-a-node"]), edges: "[]" };
  const oversized = { nodes: "[" + " ".repeat(48_001) + "]", edges: "[]" };
  const extraTopLevel = { nodes: "[]", edges: "[]", unexpected: "retain-for-validation" };
  const nestedProperty = {
    nodes: JSON.stringify([{
      id: "new:1",
      label: "service",
      graphKind: "operation",
      type: "Service",
      properties: { serialized: "[\"leave\",\"as text\"]" }
    }]),
    edges: "[]"
  };

  assert.deepEqual(prepareArguments(malformed), { nodes: malformed.nodes, edges: [] });
  assert.deepEqual(prepareArguments(nonArray), { nodes: nonArray.nodes, edges: [] });
  assert.equal(Check(tool.parameters, prepareArguments(malformed)), false);
  assert.equal(Check(tool.parameters, prepareArguments(nonArray)), false);
  assert.equal(Check(tool.parameters, prepareArguments(nestedString)), false);
  assert.equal(Check(tool.parameters, prepareArguments(oversized)), false);
  assert.equal(Check(tool.parameters, prepareArguments(extraTopLevel)), false);
  const nestedPrepared = prepareArguments(nestedProperty) as { nodes: Array<{ properties?: Record<string, unknown> }> };
  assert.equal(nestedPrepared.nodes[0]?.properties?.serialized, "[\"leave\",\"as text\"]");
  assert.equal(Check(tool.parameters, nestedPrepared), true);
});

test("planner terminal tool exposes discriminated command schemas", () => {
  const tool = createPlannerSubmitTool();
  const schema = tool.parameters as unknown as {
    properties: {
      commands: {
        anyOf?: Array<{
          type?: string;
          maxItems?: number;
          items?: {
            anyOf?: Array<{
              additionalProperties?: boolean;
              required?: string[];
              properties?: {
                kind?: { const?: string };
                type?: unknown;
                expectedVersion?: unknown;
                tasks?: {
                  items?: {
                    required?: string[];
                    properties?: {
                      id?: { pattern?: string };
                      scopeRef?: { pattern?: string };
                    };
                  };
                };
              };
            }>;
          };
        }>;
      };
      reason?: unknown;
      basedOnRefs?: unknown;
    };
    required?: string[];
    additionalProperties?: boolean;
  };
  const commandArray = schema.properties.commands.anyOf?.find((candidate) => candidate.type === "array");
  const branches = commandArray?.items?.anyOf ?? [];

  assert.equal(commandArray?.maxItems, 32);
  assert.deepEqual(
    branches.map((branch) => branch.properties?.kind?.const),
    ["create_tasks", "patch_task", "replace_dependencies", "set_task_status", "set_node_status"]
  );
  assert.ok(branches.every((branch) => branch.additionalProperties === false));
  assert.ok(branches.every((branch) => branch.required?.includes("kind")));
  assert.ok(branches.every((branch) => branch.properties?.type === undefined));
  assert.ok(branches.every((branch) => branch.properties?.expectedVersion === undefined));
  assert.equal(branches[0]?.properties?.tasks?.items?.properties?.id?.pattern, "^task:.+");
  assert.equal(branches[0]?.properties?.tasks?.items?.properties?.scopeRef?.pattern, "^scope:.+");
  assert.equal(schema.required?.includes("reason"), false);
  assert.equal(schema.required?.includes("basedOnRefs"), false);
  assert.equal(schema.additionalProperties, false);
});

test("planner terminal tool validates graph semantics before terminating", async () => {
  let validated = false;
  const tool = createPlannerSubmitTool({
    validate: () => {
      validated = true;
      throw new Error("Dependency graph would contain a cycle: task:a -> task:b -> task:a");
    }
  });

  await assert.rejects(
    () => tool.execute(
      "call:planner",
      {
        decision: "apply_commands",
        commands: [],
        reason: "invalid dependency update",
        basedOnRefs: []
      },
      new AbortController().signal,
      () => undefined,
      {} as never
    ),
    /task:a -> task:b -> task:a/
  );
  assert.equal(validated, true);
});
