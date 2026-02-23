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
      "Take over LLM control for a call session. After takeover, you will receive caller transcripts and must provide replies.",
    parameters: Type.Object({
      session_id: Type.String({ description: "Session ID to take over" }),
    }),
    async execute(_id: string, params: { session_id: string }) {
      const ok = await bridge.takeover(params.session_id);
      return {
        content: [
          { type: "text", text: ok ? "Takeover active." : "Takeover failed." },
        ],
        details: { ok },
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
            description: "Scope: global, session, or turn (default: global)",
            default: "global",
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
      bridge.sendRaw({
        type: "set_instructions",
        instructions: params.text,
        scope: params.scope || "global",
        session_id: params.session_id || "",
      });
      return {
        content: [{ type: "text", text: "Instructions set." }],
        details: { ok: true },
      };
    },
  });

  // --- Slash command ---

  api.registerCommand({
    name: "clawfinger",
    description: "Clawfinger gateway status and quick actions.",
    acceptsArgs: true,
    handler: async (ctx: { args?: string }) => {
      const args = ctx.args?.trim() || "";
      const tokens = args.split(/\s+/).filter(Boolean);
      const action = (tokens[0] || "status").toLowerCase();

      if (action === "status") {
        try {
          const s = await client.status();
          const bridgeOk = bridge.isConnected ? "connected" : "disconnected";
          return {
            text: [
              `Gateway: ${s.mlx_audio?.ok ? "healthy" : "degraded"}`,
              `Bridge: ${bridgeOk}`,
              `Sessions: ${s.active_sessions || 0}`,
              `Uptime: ${Math.floor((s.uptime_s || 0) / 60)}m`,
            ].join("\n"),
          };
        } catch (e) {
          return { text: `Gateway unreachable: ${e}` };
        }
      }

      if (action === "dial" && tokens[1]) {
        const result = await client.dial(tokens[1]);
        return {
          text: result.ok
            ? `Dialing ${tokens[1]}...`
            : `Dial failed: ${result.detail}`,
        };
      }

      return {
        text: [
          "Clawfinger commands:",
          "",
          "/clawfinger status",
          "/clawfinger dial <number>",
        ].join("\n"),
      };
    },
  });
}
