// The only tools pi gets in the paper runs. They call the local knowledge service, which already knows the
// case's date and company and enforces the reading limit, so nothing here can reach past the decision date.
import { type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const url = "http://127.0.0.1:8098";

async function ask(path: string, body: object) {
  const r = await fetch(`${url}/${path}`, { method: "POST", body: JSON.stringify(body) });
  const text = await r.text();
  return { content: [{ type: "text" as const, text }], details: {} };
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "kb_search",
    label: "Search knowledge base",
    description: "Search past information. scope: own (this company's earlier calls), similar (other companies' calls and what happened next), handbook (general analyst notes). Returns document ids with a short preview.",
    parameters: Type.Object({
      query: Type.String({ description: "Words to search for" }),
      scope: Type.String({ description: "own, similar or handbook" }),
    }),
    async execute(_id, params) {
      return ask("search", params);
    },
  });
  pi.registerTool({
    name: "kb_read",
    label: "Read a document",
    description: "Read one document by id from kb_search, or the id facts for this company's recent quarterly figures. Reading is limited, so choose what matters.",
    parameters: Type.Object({ id: Type.String({ description: "Document id, or facts" }) }),
    async execute(_id, params) {
      return ask("read", params);
    },
  });
}
