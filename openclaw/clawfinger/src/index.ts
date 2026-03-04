/**
 * Clawfinger OpenClaw plugin entry point.
 *
 * Registers a background WS bridge service, LLM-callable tools for
 * call control/observation, and a /clawfinger slash command.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import { GatewayClient } from "./gateway-client.js";
import { WsBridge } from "./ws-bridge.js";

export default function register(api: OpenClawPluginApi) {
  const cfg = api.pluginConfig as
    | { gatewayUrl?: string; bearerToken?: string }
    | undefined;
  const gatewayUrl = cfg?.gatewayUrl || "http://127.0.0.1:8996";
  const bearerToken = cfg?.bearerToken || "";

  const client = new GatewayClient(gatewayUrl, bearerToken);
  const bridge = new WsBridge(gatewayUrl, api.logger);

  // --- Background service: persistent WS bridge ---

  api.registerService({
    id: "clawfinger-bridge",
    start: async () => {
      await bridge.connect();
      api.logger.info(`Clawfinger bridge connected to ${gatewayUrl}`);
    },
    stop: async () => {
      await bridge.disconnect();
      api.logger.info("Clawfinger bridge disconnected");
    },
  });

  // --- Tools (available to LLM agents) ---

  api.registerTool({
    name: "clawfinger_status",
    label: "Clawfinger Status",
    description:
      "Check Clawfinger gateway health, active sessions, and bridge connection status.",
    parameters: Type.Object({}),
    async execute() {
      const status = await client.status();
      return {
        content: [{ type: "text", text: JSON.stringify(status) }],
        details: status,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_sessions",
    label: "Clawfinger Sessions",
    description: "List active call sessions on the Clawfinger gateway.",
    parameters: Type.Object({}),
    async execute() {
      const sessions = await client.getSessions();
      return {
        content: [{ type: "text", text: JSON.stringify(sessions) }],
        details: { sessions },
      };
    },
  });

  api.registerTool({
    name: "clawfinger_call_state",
    label: "Clawfinger Call State",
    description:
      "Get full call state for a session: conversation history, instructions, takeover status.",
    parameters: Type.Object({
      session_id: Type.String({ description: "Session ID" }),
    }),
    async execute(_id: string, params: { session_id: string }) {
      const state = await client.getCallState(params.session_id);
      return {
        content: [{ type: "text", text: JSON.stringify(state) }],
        details: state,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_dial",
    label: "Clawfinger Dial",
    description:
      "Dial an outbound phone call. The phone must be connected via ADB.",
    parameters: Type.Object({
      number: Type.String({
        description: "Phone number to dial (e.g., +49123456789)",
      }),
    }),
    async execute(_id: string, params: { number: string }) {
      const result = await client.dial(params.number);
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_hangup",
    label: "Clawfinger Hangup",
    description:
      "Force hang up the active phone call via ADB and end the gateway session.",
    parameters: Type.Object({
      session_id: Type.Optional(
        Type.String({
          description:
            "Session ID to end (optional — auto-detects if only one active)",
        }),
      ),
    }),
    async execute(
      _id: string,
      params: { session_id?: string },
    ) {
      const result = await client.hangup(params.session_id);
      return {
        content: [
          {
            type: "text",
            text: result.ok
              ? "Call hung up."
              : `Hangup failed: ${JSON.stringify(result)}`,
          },
        ],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_inject",
    label: "Clawfinger Inject TTS",
    description:
      "Inject a TTS message into the active call. The text is synthesized and played to the caller.",
    parameters: Type.Object({
      text: Type.String({ description: "Text to synthesize and play" }),
      session_id: Type.Optional(
        Type.String({ description: "Session ID (optional)" }),
      ),
    }),
    async execute(
      _id: string,
      params: { text: string; session_id?: string },
    ) {
      const result = await client.inject(params.text, params.session_id);
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_takeover",
    label: "Clawfinger Takeover",
    description:
      "Take over LLM control for a call session. After takeover, use clawfinger_turn_wait to receive caller transcripts, then clawfinger_turn_reply to respond. Call clawfinger_release when done.",
    parameters: Type.Object({
      session_id: Type.String({ description: "Session ID to take over" }),
    }),
    async execute(_id: string, params: { session_id: string }) {
      const ok = await bridge.takeover(params.session_id);
      return {
        content: [
          { type: "text", text: ok ? "Takeover active. Use clawfinger_turn_wait to receive the next caller turn." : "Takeover failed." },
        ],
        details: { ok },
      };
    },
  });

  api.registerTool({
    name: "clawfinger_turn_wait",
    label: "Clawfinger Wait for Turn",
    description:
      "Wait for the next caller turn during takeover. Returns the caller's transcript and a request_id. You MUST then call clawfinger_turn_reply with that request_id and your response text. Times out after 30 seconds if no turn arrives.",
    parameters: Type.Object({
      timeout_ms: Type.Optional(
        Type.Number({ description: "Timeout in ms (default: 30000)", default: 30000 }),
      ),
    }),
    async execute(_id: string, params: { timeout_ms?: number }) {
      const turn = await bridge.popTurnRequest(params.timeout_ms || 30000);
      if (!turn) {
        return {
          content: [
            { type: "text", text: "No turn arrived within timeout. The caller may have hung up or is still speaking. You can call clawfinger_turn_wait again to keep waiting, or clawfinger_release to hand back control." },
          ],
        };
      }
      return {
        content: [
          { type: "text", text: `Caller said: "${turn.transcript}"\n\nrequest_id: ${turn.request_id}\n\nYou MUST now call clawfinger_turn_reply with this request_id and your response.` },
        ],
        details: turn,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_turn_reply",
    label: "Clawfinger Turn Reply",
    description:
      "Send your reply to the caller and wait for their next turn. Returns the next caller transcript + request_id (same as turn_wait). If no next turn arrives within 45s, returns a timeout notice. Call this in a loop for multi-turn conversations.",
    parameters: Type.Object({
      request_id: Type.String({ description: "The request_id from clawfinger_turn_wait or previous turn_reply" }),
      reply: Type.String({ description: "Your response text (will be spoken to the caller via TTS)" }),
    }),
    async execute(_id: string, params: { request_id: string; reply: string }) {
      bridge.sendTurnReply(params.request_id, params.reply);

      // Immediately start waiting for the next turn — eliminates the
      // LLM think-time gap between turn_reply and the next turn_wait
      const next = await bridge.popTurnRequest(45_000);
      if (!next) {
        return {
          content: [
            { type: "text", text: `Reply sent: "${params.reply}"\n\nNo next turn arrived within 45s. The caller may have hung up. Call clawfinger_turn_wait to keep waiting, or clawfinger_release to hand back control.` },
          ],
        };
      }
      return {
        content: [
          { type: "text", text: `Reply sent: "${params.reply}"\n\nNext turn — Caller said: "${next.transcript}"\n\nrequest_id: ${next.request_id}\n\nCall clawfinger_turn_reply again with this request_id and your response.` },
        ],
        details: next,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_release",
    label: "Clawfinger Release",
    description:
      "Release LLM control for a call session back to the local gateway LLM.",
    parameters: Type.Object({
      session_id: Type.String({ description: "Session ID to release" }),
    }),
    async execute(_id: string, params: { session_id: string }) {
      const ok = await bridge.release(params.session_id);
      return {
        content: [
          { type: "text", text: ok ? "Released." : "Release failed." },
        ],
        details: { ok },
      };
    },
  });

  api.registerTool({
    name: "clawfinger_context_set",
    label: "Clawfinger Set Context",
    description:
      "Inject knowledge into a call session. The LLM sees this as context before each user turn. Replaces any existing context.",
    parameters: Type.Object({
      session_id: Type.String({ description: "Session ID" }),
      context: Type.String({ description: "Knowledge text to inject" }),
    }),
    async execute(
      _id: string,
      params: { session_id: string; context: string },
    ) {
      const result = await client.setContext(
        params.session_id,
        params.context,
      );
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_context_clear",
    label: "Clawfinger Clear Context",
    description: "Clear injected knowledge from a call session.",
    parameters: Type.Object({
      session_id: Type.String({ description: "Session ID" }),
    }),
    async execute(_id: string, params: { session_id: string }) {
      const result = await client.clearContext(params.session_id);
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_call_config_get",
    label: "Clawfinger Get Call Config",
    description:
      "Read current call policy settings: auto-answer, greetings, caller filtering, max duration, auth.",
    parameters: Type.Object({}),
    async execute() {
      const config = await client.getCallConfig();
      return {
        content: [{ type: "text", text: JSON.stringify(config) }],
        details: config,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_call_config_set",
    label: "Clawfinger Set Call Config",
    description:
      "Update call policy settings. Pass only the fields you want to change.",
    parameters: Type.Object({
      config: Type.Record(Type.String(), Type.Unknown(), {
        description: "Config fields to update",
      }),
    }),
    async execute(
      _id: string,
      params: { config: Record<string, unknown> },
    ) {
      const result = await client.setCallConfig(params.config);
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_session_end",
    label: "Clawfinger End Session",
    description:
      "Mark a call session as ended (hung up). Moves it from active to ended state.",
    parameters: Type.Object({
      session_id: Type.String({ description: "Session ID to end" }),
    }),
    async execute(_id: string, params: { session_id: string }) {
      const result = await client.endSession(params.session_id);
      return {
        content: [
          {
            type: "text",
            text: result.ok
              ? `Session ${params.session_id} ended.`
              : `Failed to end session: ${JSON.stringify(result)}`,
          },
        ],
        details: result,
      };
    },
  });

  // --- Robot tools ---

  api.registerTool({
    name: "clawfinger_robot_status",
    label: "Clawfinger Robot Status",
    description:
      "Get robot status: connection state, transport info, model, capabilities. Works even when robot is disconnected — shows transport state.",
    parameters: Type.Object({}),
    async execute() {
      const config = await client.getRobotConfig();
      return {
        content: [{ type: "text", text: JSON.stringify(config) }],
        details: config,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_command",
    label: "Clawfinger Robot Command",
    description:
      "Send a command to the connected robot via Intercom transport. The robot must be connected (check clawfinger_robot_status first). Fire-and-forget commands (stop, safety_stop) return immediately; others wait for the robot's response.",
    parameters: Type.Object({
      command: Type.String({
        description: "Command type (walk, stop, stand, sit, look, speak, pick_up, etc.)",
      }),
      params: Type.Optional(
        Type.Record(Type.String(), Type.Unknown(), {
          description: "Command parameters (e.g., {speed: 0.3, direction: 'forward'})",
        }),
      ),
    }),
    async execute(
      _id: string,
      params: { command: string; params?: Record<string, unknown> },
    ) {
      // Send via WS bridge (robot_command message type)
      const ackPromise = bridge.waitForAck("robot.command.ack", 15000);
      bridge.sendRaw({
        type: "robot_command",
        command: {
          type: params.command,
          params: params.params || {},
        },
      });
      const ack = await ackPromise;
      if (!ack) {
        return {
          content: [{ type: "text", text: "Robot command timed out — no response from gateway." }],
          details: { ok: false, error: "timeout" },
        };
      }
      return {
        content: [{
          type: "text",
          text: ack.ok
            ? `Command '${params.command}' sent. ${ack.detail || ""}`
            : `Command failed: ${ack.error || "unknown error"}`,
        }],
        details: ack,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_config_get",
    label: "Clawfinger Get Robot Config",
    description:
      "Read robot configuration: model, transport, capabilities, and model-specific settings.",
    parameters: Type.Object({}),
    async execute() {
      const config = await client.getRobotConfig();
      return {
        content: [{ type: "text", text: JSON.stringify(config) }],
        details: config,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_config_set",
    label: "Clawfinger Set Robot Config",
    description:
      "Update robot configuration. Pass only fields to change. Security keys (intercom_key, safety_stop_on_disconnect, enable_low_level) are blocked from agents.",
    parameters: Type.Object({
      config: Type.Record(Type.String(), Type.Unknown(), {
        description: "Config fields to update",
      }),
    }),
    async execute(
      _id: string,
      params: { config: Record<string, unknown> },
    ) {
      const result = await client.setRobotConfig(params.config);
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
        details: result,
      };
    },
  });

  // --- Robot skill + project tools ---

  api.registerTool({
    name: "clawfinger_robot_skill_list",
    label: "Clawfinger Robot Skills",
    description:
      "List available robot skills (slow-path LLM knowledge and fast-path trained policies).",
    parameters: Type.Object({}),
    async execute() {
      const skills = await client.listRobotSkills();
      return {
        content: [{ type: "text", text: JSON.stringify(skills, null, 2) }],
        details: { skills },
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_skill_topic",
    label: "Clawfinger Robot Skill Topic",
    description:
      "Read a robot skill topic's knowledge content (e.g., household_objects/common).",
    parameters: Type.Object({
      name: Type.String({ description: "Skill name (e.g., household_objects)" }),
      topic: Type.String({ description: "Topic name (e.g., common)" }),
    }),
    async execute(_id: string, params: { name: string; topic: string }) {
      const result = await client.getRobotSkillTopic(params.name, params.topic);
      return {
        content: [{ type: "text", text: result.content || JSON.stringify(result) }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_project_status",
    label: "Clawfinger Robot Project Status",
    description:
      "Get current robot project execution state including structured plan with step status (pending/active/completed/failed/skipped), dependencies, verification results, and progress summary.",
    parameters: Type.Object({}),
    async execute() {
      const project = await client.getRobotProjectStatus();
      return {
        content: [{ type: "text", text: JSON.stringify(project, null, 2) }],
        details: project,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_project_cancel",
    label: "Clawfinger Robot Cancel Project",
    description:
      "Cancel the currently running robot project.",
    parameters: Type.Object({}),
    async execute() {
      const result = await client.cancelRobotProject();
      return {
        content: [{ type: "text", text: result.message || JSON.stringify(result) }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_takeover",
    label: "Clawfinger Robot Takeover",
    description:
      "Take full control of the robot endpoint. User voice → transcript forwarded to you. Your replies are spoken on the robot speaker. Use clawfinger_robot_turn_wait and clawfinger_robot_turn_reply for the conversation loop.",
    parameters: Type.Object({}),
    async execute() {
      const ok = await bridge.robotTakeover();
      return {
        content: [
          { type: "text", text: ok ? "Robot takeover active. Use clawfinger_robot_turn_wait to receive voice input." : "Robot takeover failed." },
        ],
        details: { ok },
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_turn_wait",
    label: "Clawfinger Robot Turn Wait",
    description:
      "Wait for user to speak to the robot during takeover. Returns transcript + request_id. You MUST call clawfinger_robot_turn_reply with this request_id.",
    parameters: Type.Object({
      timeout_ms: Type.Optional(
        Type.Number({ description: "Timeout in ms (default: 30000)", default: 30000 }),
      ),
    }),
    async execute(_id: string, params: { timeout_ms?: number }) {
      const turn = await bridge.popRobotTurnRequest(params.timeout_ms || 30000);
      if (!turn) {
        return {
          content: [
            { type: "text", text: "No robot voice input within timeout. Call clawfinger_robot_turn_wait again or clawfinger_robot_release." },
          ],
        };
      }
      return {
        content: [
          { type: "text", text: `User said to robot: "${turn.transcript}"\n\nrequest_id: ${turn.request_id}\n\nCall clawfinger_robot_turn_reply with this request_id and your response.` },
        ],
        details: turn,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_turn_reply",
    label: "Clawfinger Robot Turn Reply",
    description:
      "Send text reply (spoken on robot speaker) + optional robot commands. Then waits for the next voice input.",
    parameters: Type.Object({
      request_id: Type.String({ description: "request_id from turn_wait or previous turn_reply" }),
      reply: Type.String({ description: "Text to speak on robot speaker" }),
      commands: Type.Optional(
        Type.Array(
          Type.Record(Type.String(), Type.Unknown()),
          { description: "Optional robot commands to execute alongside speech" },
        ),
      ),
    }),
    async execute(
      _id: string,
      params: { request_id: string; reply: string; commands?: Array<Record<string, unknown>> },
    ) {
      bridge.sendRobotTurnReply(params.request_id, params.reply, params.commands);

      const next = await bridge.popRobotTurnRequest(45_000);
      if (!next) {
        return {
          content: [
            { type: "text", text: `Reply sent: "${params.reply}"\n\nNo next voice input within 45s. Call clawfinger_robot_turn_wait or clawfinger_robot_release.` },
          ],
        };
      }
      return {
        content: [
          { type: "text", text: `Reply sent: "${params.reply}"\n\nUser said to robot: "${next.transcript}"\n\nrequest_id: ${next.request_id}\n\nCall clawfinger_robot_turn_reply again.` },
        ],
        details: next,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_release",
    label: "Clawfinger Robot Release",
    description:
      "Release robot control back to local LLM.",
    parameters: Type.Object({}),
    async execute() {
      const ok = await bridge.robotRelease();
      return {
        content: [
          { type: "text", text: ok ? "Robot released." : "Robot release failed." },
        ],
        details: { ok },
      };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_snapshot",
    label: "Clawfinger Robot Snapshot",
    description:
      "Capture a single camera frame from the robot. Returns the image so the LLM can see what the robot sees.",
    parameters: Type.Object({
      source: Type.Optional(
        Type.String({
          description:
            "Camera source ID (e.g. 'head_rgb', 'head_depth'). Defaults to first available camera.",
        }),
      ),
    }),
    async execute(
      _id: string,
      params: { source?: string },
    ) {
      const result = await client.robotSnapshot(params.source);
      if (!result.ok) {
        return {
          content: [
            {
              type: "text",
              text: `Snapshot failed: ${result.detail || "unknown error"}`,
            },
          ],
          details: result,
        };
      }
      const content: any[] = [];
      if (result.image_base64) {
        content.push({
          type: "image",
          data: result.image_base64,
          mimeType: "image/jpeg",
        });
      }
      content.push({
        type: "text",
        text: `Snapshot captured from ${result.source || "camera"} (${result.width}x${result.height}).`,
      });
      return { content, details: { ok: true, source: result.source } };
    },
  });

  api.registerTool({
    name: "clawfinger_robot_describe",
    label: "Clawfinger Robot Describe Scene",
    description:
      "Capture a camera frame and run VLM scene description on the robot's Jetson. Returns the description text and optionally the image.",
    parameters: Type.Object({
      source: Type.Optional(
        Type.String({
          description: "Camera source ID. Defaults to first available camera.",
        }),
      ),
      prompt: Type.Optional(
        Type.String({
          description:
            "Prompt for the VLM (default: 'Describe what you see.')",
        }),
      ),
    }),
    async execute(
      _id: string,
      params: { source?: string; prompt?: string },
    ) {
      const result = await client.robotDescribe(params.source, params.prompt);
      if (!result.ok) {
        return {
          content: [
            {
              type: "text",
              text: `Describe failed: ${result.detail || "unknown error"}`,
            },
          ],
          details: result,
        };
      }
      const content: any[] = [];
      if (result.image_base64) {
        content.push({
          type: "image",
          data: result.image_base64,
          mimeType: "image/jpeg",
        });
      }
      content.push({
        type: "text",
        text: result.description || "(no description returned)",
      });
      return {
        content,
        details: { ok: true, source: result.source },
      };
    },
  });

  api.registerTool({
    name: "clawfinger_instructions_set",
    label: "Clawfinger Set Instructions",
    description:
      "Set the LLM system instructions. Scope: 'global' (all sessions), 'session' (one session), or 'turn' (consumed after one turn).",
    parameters: Type.Object({
      text: Type.String({ description: "Instruction text" }),
      scope: Type.Optional(
        Type.Union(
          [
            Type.Literal("global"),
            Type.Literal("session"),
            Type.Literal("turn"),
          ],
          {
            description: "Scope: session or turn (default: session). Global scope is disabled.",
            default: "session",
          },
        ),
      ),
      session_id: Type.Optional(
        Type.String({
          description: "Session ID (required for session/turn scope)",
        }),
      ),
    }),
    async execute(
      _id: string,
      params: { text: string; scope?: string; session_id?: string },
    ) {
      const scope = params.scope === "turn" ? "turn" : "session";
      if (!params.session_id) {
        return {
          content: [{ type: "text", text: "Error: session_id is required." }],
          details: { ok: false },
        };
      }
      bridge.sendRaw({
        type: "set_instructions",
        instructions: params.text,
        scope,
        session_id: params.session_id,
      });
      return {
        content: [{ type: "text", text: "Instructions set." }],
        details: { ok: true },
      };
    },
  });

  // --- Spatial memory tools ---

  api.registerTool({
    name: "clawfinger_memory_teach_person",
    label: "Clawfinger Teach Person",
    description: "Teach the robot to recognize a person by name and optional description.",
    parameters: Type.Object({
      name: Type.String({ description: "Person name" }),
      description: Type.Optional(Type.String({ description: "Description of the person" })),
    }),
    async execute(_id: string, params: { name: string; description?: string }) {
      const result = await client.memoryAddPerson(params.name, params.description);
      return {
        content: [{ type: "text", text: `Person added: ${result.name} (id: ${result.id})` }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_memory_teach_object",
    label: "Clawfinger Teach Object",
    description: "Teach the robot to recognize an object by name and optional description.",
    parameters: Type.Object({
      name: Type.String({ description: "Object name" }),
      description: Type.Optional(Type.String({ description: "Description of the object" })),
    }),
    async execute(_id: string, params: { name: string; description?: string }) {
      const result = await client.memoryAddObject(params.name, params.description);
      return {
        content: [{ type: "text", text: `Object added: ${result.name} (id: ${result.id})` }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_memory_teach_room",
    label: "Clawfinger Teach Room",
    description: "Define a room or zone for spatial memory.",
    parameters: Type.Object({
      name: Type.String({ description: "Room name" }),
      description: Type.Optional(Type.String({ description: "Description of the room" })),
    }),
    async execute(_id: string, params: { name: string; description?: string }) {
      const result = await client.memoryAddRoom(params.name, params.description);
      return {
        content: [{ type: "text", text: `Room added: ${result.name} (id: ${result.id})` }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_memory_query",
    label: "Clawfinger Memory Query",
    description: "Search spatial memory with natural language. Supports text search, person sightings, object sightings, room activity, and nearby queries.",
    parameters: Type.Object({
      type: Type.Union([
        Type.Literal("text"),
        Type.Literal("person_sightings"),
        Type.Literal("object_sightings"),
        Type.Literal("room_activity"),
        Type.Literal("nearby"),
      ], { description: "Query type", default: "text" }),
      text: Type.Optional(Type.String({ description: "Search text (for text queries)" })),
      person_id: Type.Optional(Type.String({ description: "Person ID (for person_sightings)" })),
      object_id: Type.Optional(Type.String({ description: "Object ID (for object_sightings)" })),
      room: Type.Optional(Type.String({ description: "Room name (for room_activity or filter)" })),
      time_filter: Type.Optional(Type.String({ description: "Natural time filter: 'last hour', 'today', 'yesterday', 'last 3 days'" })),
      n_results: Type.Optional(Type.Number({ description: "Max results", default: 10 })),
    }),
    async execute(_id: string, params: any) {
      const body: any = { type: params.type, n_results: params.n_results || 10 };
      if (params.text) body.text = params.text;
      if (params.person_id) body.person_id = params.person_id;
      if (params.object_id) body.object_id = params.object_id;
      if (params.room) body.room = params.room;
      if (params.time_filter) body.time_filter = params.time_filter;
      if (params.room && params.type !== "room_activity") {
        body.filters = { room: params.room };
      }
      const result = await client.memoryQuery(params.type, body);
      const lines = [`Found ${result.count} results:`];
      for (const r of (result.results || []).slice(0, 10)) {
        const timeStr = r.time_ago ? ` (${r.time_ago})` : "";
        lines.push(`  [${r.entity_type || "scene"}] ${r.entity_name || "--"} in ${r.room || "--"}: ${(r.description || r.document || "").slice(0, 80)}${timeStr}`);
      }
      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: result,
      };
    },
  });

  api.registerTool({
    name: "clawfinger_memory_last_seen",
    label: "Clawfinger Memory Last Seen",
    description: "Find the most recent observation of a person, object, or activity in a room.",
    parameters: Type.Object({
      entity_name: Type.Optional(Type.String({ description: "Name of person or object" })),
      entity_type: Type.Optional(Type.String({ description: "Entity type: person, object" })),
      room: Type.Optional(Type.String({ description: "Room name" })),
    }),
    async execute(_id: string, params: any) {
      const result = await client.memoryLastSeen(params);
      if (!result.found) {
        return { content: [{ type: "text", text: "No matching observation found." }], details: result };
      }
      const r = result.result;
      const text = `Last seen: ${r.entity_name || r.entity_type || "entity"} in ${r.room || "unknown"} — ${r.time_ago || "unknown time"}`;
      return { content: [{ type: "text", text }], details: result };
    },
  });

  api.registerTool({
    name: "clawfinger_memory_list",
    label: "Clawfinger Memory List",
    description: "List known entities by type (persons, objects, rooms, routines).",
    parameters: Type.Object({
      entity_type: Type.Union([
        Type.Literal("persons"),
        Type.Literal("objects"),
        Type.Literal("rooms"),
        Type.Literal("routines"),
      ], { description: "Entity type to list" }),
    }),
    async execute(_id: string, params: { entity_type: string }) {
      let items: any[];
      if (params.entity_type === "persons") items = await client.memoryListPersons();
      else if (params.entity_type === "objects") items = await client.memoryListObjects();
      else if (params.entity_type === "rooms") items = await client.memoryListRooms();
      else items = [];
      if (!items.length) return { content: [{ type: "text", text: `No ${params.entity_type} found.` }], details: { count: 0 } };
      const lines = items.map((i: any) => `  ${i.name || i.id}${i.description ? ` — ${i.description}` : ""}${i.schedule ? ` (${i.schedule})` : ""}`);
      return {
        content: [{ type: "text", text: `${params.entity_type} (${items.length}):\n${lines.join("\n")}` }],
        details: { count: items.length, items },
      };
    },
  });

  api.registerTool({
    name: "clawfinger_memory_stats",
    label: "Clawfinger Memory Stats",
    description: "Get spatial memory database statistics.",
    parameters: Type.Object({}),
    async execute() {
      const stats = await client.memoryStats();
      const lines = [
        `Initialized: ${stats.initialized}`,
        `Persons: ${stats.persons || 0}`,
        `Objects: ${stats.objects || 0}`,
        `Rooms: ${stats.rooms || 0}`,
        `Routines: ${stats.routines || 0}`,
        `Observations: ${stats.observations || 0}`,
        `CLIP loaded: ${stats.clip_loaded || false}`,
      ];
      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: stats,
      };
    },
  });

  // --- Slash command ---

  const HELP_TEXT = [
    "Clawfinger commands:",
    "",
    "/clawfinger                                  — this help",
    "/clawfinger status                           — gateway health, bridge, sessions, uptime",
    "/clawfinger sessions                         — list active session IDs",
    "/clawfinger state <session_id>               — full call state (history, instructions, takeover)",
    "/clawfinger dial <number>                    — dial outbound call (e.g. +49123456789)",
    "/clawfinger hangup [session_id]              — force hang up the active call",
    "/clawfinger inject <text>                    — inject TTS into active call (first session)",
    "/clawfinger inject <session_id> <text>       — inject TTS into specific session",
    "/clawfinger takeover <session_id>            — take over LLM control for a session",
    "/clawfinger release <session_id>             — release LLM control back to local LLM",
    "/clawfinger context get <session_id>         — read injected knowledge",
    "/clawfinger context set <session_id> <text>  — inject/replace knowledge",
    "/clawfinger context clear <session_id>       — clear injected knowledge",
    "/clawfinger config call                      — show call policy settings",
    "/clawfinger config tts                       — show TTS voice and speed",
    "/clawfinger config llm                       — show LLM model and params",
    "/clawfinger config robot                     — show robot config and capabilities",
    "/clawfinger instructions <text>              — set global LLM instructions",
    "/clawfinger instructions <session_id> <text> — set per-session instructions",
    "/clawfinger end <session_id>                 — mark a session as ended (hung up)",
    "/clawfinger robot status                     — robot config, connection, transport state",
    "/clawfinger robot command <type> [params]    — send robot command (e.g. walk, stop, look)",
    "/clawfinger robot skills                     — list robot skills (slow + fast)",
    "/clawfinger robot skill <name> <topic>       — read skill topic content",
    "/clawfinger robot project                    — current project status",
    "/clawfinger robot project cancel             — cancel running project",
    "/clawfinger robot takeover                   — take control of robot",
    "/clawfinger robot release                    — release robot control",
    "/clawfinger robot perception                 — list perception sources (cameras, mics)",
    "/clawfinger robot snapshot [source]          — capture camera snapshot",
    "/clawfinger robot describe [source] [prompt] — VLM scene description",
    "/clawfinger robot stream start|stop [source] — video stream control",
    "/clawfinger robot audio start|stop [source]  — audio monitor control",
    "/clawfinger robot memory stats               — spatial memory stats",
    "/clawfinger robot memory persons             — list known persons",
    "/clawfinger robot memory objects             — list known objects",
    "/clawfinger robot memory rooms               — list known rooms",
    "/clawfinger robot memory query <text>        — search spatial memory",
  ].join("\n");

  api.registerCommand({
    name: "clawfinger",
    description: "Clawfinger gateway control — status, dial, inject, takeover, context, config.",
    acceptsArgs: true,
    handler: async (ctx: { args?: string }) => {
      const args = ctx.args?.trim() || "";
      const tokens = args.split(/\s+/).filter(Boolean);
      const action = (tokens[0] || "help").toLowerCase();

      try {
        // --- status ---
        if (action === "status") {
          const s = await client.status();
          const bridgeOk = bridge.isConnected ? "connected" : "disconnected";
          const agents = s.agents?.length || 0;
          const takenOver = bridge.takenOverSessions.size;
          return {
            text: [
              `Gateway: ${s.mlx_audio?.ok ? "healthy" : "degraded"}`,
              `Bridge: ${bridgeOk}`,
              `Sessions: ${s.active_sessions || 0}`,
              `Agents: ${agents}`,
              `Takeovers: ${takenOver}`,
              `Uptime: ${Math.floor((s.uptime_s || 0) / 60)}m`,
              `LLM: ${s.llm?.model || "unknown"} (${s.llm?.loaded ? "loaded" : "not loaded"})`,
            ].join("\n"),
          };
        }

        // --- sessions ---
        if (action === "sessions") {
          const sessions = await client.getSessions();
          if (!sessions.length) return { text: "No active sessions." };
          return { text: `Active sessions (${sessions.length}):\n${sessions.map((s: string) => `  ${s}`).join("\n")}` };
        }

        // --- state <session_id> ---
        if (action === "state") {
          if (!tokens[1]) return { text: "Usage: /clawfinger state <session_id>" };
          const state = await client.getCallState(tokens[1]);
          const lines = [
            `Session: ${state.session_id}`,
            `Turns: ${state.turn_count}`,
            `Takeover: ${state.agent_takeover ? "yes" : "no"}`,
          ];
          if (state.history?.length) {
            lines.push("", "Recent:");
            for (const msg of state.history.slice(-4)) {
              const preview = String(msg.content || "").slice(0, 80);
              lines.push(`  ${msg.role}: ${preview}`);
            }
          }
          return { text: lines.join("\n") };
        }

        // --- dial <number> ---
        if (action === "dial") {
          if (!tokens[1]) return { text: "Usage: /clawfinger dial <number>" };
          const result = await client.dial(tokens[1]);
          return { text: result.ok ? `Dialing ${tokens[1]}...` : `Dial failed: ${result.detail}` };
        }

        // --- hangup [session_id] ---
        if (action === "hangup") {
          const result = await client.hangup(tokens[1]);
          return { text: result.ok ? `Call hung up.${result.session_id ? ` Session: ${result.session_id}` : ''}` : `Hangup failed: ${JSON.stringify(result)}` };
        }

        // --- inject [session_id] <text> ---
        if (action === "inject") {
          if (!tokens[1]) return { text: "Usage: /clawfinger inject <text>  or  /clawfinger inject <session_id> <text>" };
          // If first arg looks like a session ID (hex, 20+ chars) and there's more text, use it as session_id
          let sid: string | undefined;
          let text: string;
          if (tokens[1].length >= 20 && /^[a-f0-9]+$/i.test(tokens[1]) && tokens[2]) {
            sid = tokens[1];
            text = tokens.slice(2).join(" ");
          } else {
            text = tokens.slice(1).join(" ");
          }
          const result = await client.inject(text, sid);
          return { text: result.ok ? `Injected: "${text}"` : `Inject failed: ${JSON.stringify(result)}` };
        }

        // --- takeover <session_id> ---
        if (action === "takeover") {
          if (!tokens[1]) return { text: "Usage: /clawfinger takeover <session_id>" };
          const ok = await bridge.takeover(tokens[1]);
          return { text: ok ? `Takeover active for ${tokens[1]}` : `Takeover failed for ${tokens[1]}` };
        }

        // --- release <session_id> ---
        if (action === "release") {
          if (!tokens[1]) return { text: "Usage: /clawfinger release <session_id>" };
          const ok = await bridge.release(tokens[1]);
          return { text: ok ? `Released ${tokens[1]}` : `Release failed for ${tokens[1]}` };
        }

        // --- end <session_id> ---
        if (action === "end") {
          if (!tokens[1]) return { text: "Usage: /clawfinger end <session_id>" };
          const result = await client.endSession(tokens[1]);
          return { text: result.ok ? `Session ${tokens[1]} ended.` : `End failed: ${JSON.stringify(result)}` };
        }

        // --- context get|set|clear <session_id> [text] ---
        if (action === "context") {
          const sub = (tokens[1] || "").toLowerCase();
          const sid = tokens[2] || "";

          if (sub === "get" && sid) {
            const ctx = await client.getContext(sid);
            return { text: ctx.has_knowledge ? `Context for ${sid}:\n${ctx.knowledge}` : `No context for ${sid}.` };
          }
          if (sub === "set" && sid && tokens[3]) {
            const text = tokens.slice(3).join(" ");
            await client.setContext(sid, text);
            return { text: `Context set for ${sid}.` };
          }
          if (sub === "clear" && sid) {
            await client.clearContext(sid);
            return { text: `Context cleared for ${sid}.` };
          }
          return { text: "Usage:\n  /clawfinger context get <session_id>\n  /clawfinger context set <session_id> <text>\n  /clawfinger context clear <session_id>" };
        }

        // --- config call|tts|llm ---
        if (action === "config") {
          const sub = (tokens[1] || "call").toLowerCase();
          if (sub === "call") {
            const cfg = await client.getCallConfig();
            return { text: JSON.stringify(cfg, null, 2) };
          }
          if (sub === "tts") {
            const cfg = await client.getTtsConfig();
            return { text: JSON.stringify(cfg, null, 2) };
          }
          if (sub === "llm") {
            const cfg = await client.getLlmConfig();
            return { text: JSON.stringify(cfg, null, 2) };
          }
          if (sub === "robot") {
            const cfg = await client.getRobotConfig();
            return { text: JSON.stringify(cfg, null, 2) };
          }
          return { text: "Usage: /clawfinger config call|tts|llm|robot" };
        }

        // --- robot status | robot command <type> [params] ---
        if (action === "robot") {
          const sub = (tokens[1] || "status").toLowerCase();
          if (sub === "status") {
            const cfg = await client.getRobotConfig();
            return { text: JSON.stringify(cfg, null, 2) };
          }
          if (sub === "command") {
            if (!tokens[2]) return { text: "Usage: /clawfinger robot command <type> [params_json]" };
            const cmdType = tokens[2];
            let cmdParams: Record<string, unknown> = {};
            if (tokens[3]) {
              try {
                cmdParams = JSON.parse(tokens.slice(3).join(" "));
              } catch {
                // Try key=value pairs
                cmdParams = {};
                for (const t of tokens.slice(3)) {
                  const [k, v] = t.split("=");
                  if (k && v !== undefined) {
                    cmdParams[k] = isNaN(Number(v)) ? v : Number(v);
                  }
                }
              }
            }
            const ackPromise = bridge.waitForAck("robot.command.ack", 15000);
            bridge.sendRaw({
              type: "robot_command",
              command: { type: cmdType, params: cmdParams },
            });
            const ack = await ackPromise;
            if (!ack) return { text: `Robot command '${cmdType}' timed out.` };
            return { text: ack.ok ? `OK: ${ack.detail || "sent"}` : `Failed: ${ack.error || "unknown"}` };
          }
          if (sub === "skills") {
            const skills = await client.listRobotSkills();
            if (!skills.length) return { text: "No robot skills loaded." };
            const lines = skills.map((s: any) =>
              `  ${s.name} (${s.execution_mode})${s.status === "coming_soon" ? " [coming soon]" : ""}: ${s.description}`
            );
            return { text: `Robot skills (${skills.length}):\n${lines.join("\n")}` };
          }
          if (sub === "skill") {
            if (!tokens[2] || !tokens[3]) return { text: "Usage: /clawfinger robot skill <name> <topic>" };
            const result = await client.getRobotSkillTopic(tokens[2], tokens[3]);
            return { text: result.content || `Topic not found: ${tokens[2]}/${tokens[3]}` };
          }
          if (sub === "project") {
            const subSub = (tokens[2] || "").toLowerCase();
            if (subSub === "cancel") {
              const result = await client.cancelRobotProject();
              return { text: result.message || JSON.stringify(result) };
            }
            const project = await client.getRobotProjectStatus();
            return { text: JSON.stringify(project, null, 2) };
          }
          if (sub === "takeover") {
            const ok = await bridge.robotTakeover();
            return { text: ok ? "Robot takeover active." : "Robot takeover failed." };
          }
          if (sub === "release") {
            const ok = await bridge.robotRelease();
            return { text: ok ? "Robot released." : "Robot release failed." };
          }
          if (sub === "perception") {
            const sources = await client.robotPerceptionSources();
            return { text: JSON.stringify(sources, null, 2) };
          }
          if (sub === "snapshot") {
            const source = tokens[2] || undefined;
            const result = await client.robotSnapshot(source);
            if (!result.ok) return { text: `Snapshot failed: ${result.detail || "unknown"}` };
            return { text: `Snapshot captured from ${result.source} (${result.width}x${result.height}).` };
          }
          if (sub === "describe") {
            const source = tokens[2] || undefined;
            const prompt = tokens.slice(3).join(" ") || undefined;
            const result = await client.robotDescribe(source, prompt);
            if (!result.ok) return { text: `Describe failed: ${result.detail || "unknown"}` };
            return { text: result.description || "(no description)" };
          }
          if (sub === "stream") {
            const subSub = (tokens[2] || "").toLowerCase();
            const source = tokens[3] || undefined;
            if (subSub === "start") {
              const result = await client.robotStreamStart(source);
              return { text: result.ok ? `Stream started: ${result.source}` : `Failed: ${result.detail || "error"}` };
            }
            if (subSub === "stop") {
              const result = await client.robotStreamStop(source);
              return { text: result.ok ? `Stream stopped: ${result.source}` : `Failed: ${result.detail || "error"}` };
            }
            return { text: "Usage: /clawfinger robot stream start|stop [source]" };
          }
          if (sub === "audio") {
            const subSub = (tokens[2] || "").toLowerCase();
            const source = tokens[3] || undefined;
            if (subSub === "start") {
              const result = await client.robotAudioMonitorStart(source);
              return { text: result.ok ? `Audio monitor started: ${result.source}` : `Failed: ${result.detail || "error"}` };
            }
            if (subSub === "stop") {
              const result = await client.robotAudioMonitorStop(source);
              return { text: result.ok ? `Audio monitor stopped: ${result.source}` : `Failed: ${result.detail || "error"}` };
            }
            return { text: "Usage: /clawfinger robot audio start|stop [source]" };
          }
          if (sub === "memory") {
            const memSub = (tokens[2] || "stats").toLowerCase();
            if (memSub === "stats") {
              const stats = await client.memoryStats();
              const temporal = stats.oldest_ago ? `\n  Oldest: ${stats.oldest_ago}\n  Newest: ${stats.newest_ago}\n  Span: ${stats.time_span}` : "";
              return { text: `Spatial Memory:\n  Persons: ${stats.persons || 0}\n  Objects: ${stats.objects || 0}\n  Rooms: ${stats.rooms || 0}\n  Routines: ${stats.routines || 0}\n  Observations: ${stats.observations || 0}\n  CLIP: ${stats.clip_loaded ? "loaded" : "not loaded"}${temporal}` };
            }
            if (memSub === "persons") {
              const persons = await client.memoryListPersons();
              if (!persons.length) return { text: "No persons in spatial memory." };
              return { text: `Persons (${persons.length}):\n${persons.map((p: any) => `  ${p.name}${p.description ? ` — ${p.description}` : ""} (${p.image_count || 0} photos)`).join("\n")}` };
            }
            if (memSub === "objects") {
              const objects = await client.memoryListObjects();
              if (!objects.length) return { text: "No objects in spatial memory." };
              return { text: `Objects (${objects.length}):\n${objects.map((o: any) => `  ${o.name}${o.description ? ` — ${o.description}` : ""}`).join("\n")}` };
            }
            if (memSub === "rooms") {
              const rooms = await client.memoryListRooms();
              if (!rooms.length) return { text: "No rooms in spatial memory." };
              return { text: `Rooms (${rooms.length}):\n${rooms.map((r: any) => `  ${r.name}${r.description ? ` — ${r.description}` : ""}`).join("\n")}` };
            }
            if (memSub === "query") {
              const queryText = tokens.slice(3).join(" ");
              if (!queryText) return { text: "Usage: /clawfinger robot memory query <text>" };
              const result = await client.memoryQuery("text", { text: queryText, n_results: 10 });
              if (!result.results?.length) return { text: "No results found." };
              const lines = result.results.slice(0, 10).map((r: any) => {
                const timeStr = r.time_ago ? ` (${r.time_ago})` : "";
                return `  [${r.entity_type || "scene"}] ${r.entity_name || "--"} in ${r.room || "--"}: ${(r.description || "").slice(0, 60)}${timeStr}`;
              });
              return { text: `Results (${result.count}):\n${lines.join("\n")}` };
            }
            if (memSub === "last_seen") {
              const name = tokens.slice(3).join(" ");
              if (!name) return { text: "Usage: /clawfinger robot memory last_seen <name>" };
              const result = await client.memoryLastSeen({ entity_name: name });
              if (!result.found) return { text: `No observations found for "${name}".` };
              const r = result.result;
              return { text: `Last seen: ${r.entity_name || r.entity_type || "entity"} in ${r.room || "unknown"} — ${r.time_ago || "unknown time"}` };
            }
            return { text: "Usage: /clawfinger robot memory stats|persons|objects|rooms|query|last_seen <text>" };
          }
          return { text: "Usage: /clawfinger robot status|command|skills|skill|project|takeover|release|perception|snapshot|describe|stream|audio|memory" };
        }

        // --- instructions <session_id> <text> ---
        if (action === "instructions") {
          if (!tokens[1] || !tokens[2]) return { text: "Usage: /clawfinger instructions <session_id> <text>" };
          const sid = tokens[1];
          const text = tokens.slice(2).join(" ");
          bridge.sendRaw({ type: "set_instructions", instructions: text, scope: "session", session_id: sid });
          return { text: `Instructions set for session ${sid}.` };
        }

        // --- help (default) ---
        return { text: HELP_TEXT };

      } catch (e) {
        return { text: `Error: ${e}` };
      }
    },
  });
}
