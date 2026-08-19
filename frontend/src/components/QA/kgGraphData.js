/** 從 agent observation 萃取圖資料；認得 spec_references 與 supersedes chain 兩種形狀。 */
export const graphFromObservation = (obs) => {
  if (!obs || typeof obs !== "object") return null;
  const out = obs.output || obs;
  if (out.center && (out.outgoing || out.incoming)) {
    const nodes = new Map([[out.center, { id: out.center, main: true }]]);
    const links = [];
    (out.outgoing || []).forEach((e) => {
      if (!e.canonical_id) return;
      nodes.set(e.canonical_id, { id: e.canonical_id });
      links.push({ source: out.center, target: e.canonical_id, rel: e.rel_type });
    });
    (out.incoming || []).forEach((e) => {
      if (!e.canonical_id) return;
      nodes.set(e.canonical_id, { id: e.canonical_id });
      links.push({ source: e.canonical_id, target: out.center, rel: e.rel_type });
    });
    if (links.length < 2) return null;
    return { nodes: [...nodes.values()], links };
  }
  if (Array.isArray(out.chain) && out.chain.length >= 3) {
    const nodes = out.chain.map((e) => ({ id: e.canonical_id, main: Boolean(e.is_center) }));
    const links = out.chain.slice(1).map((e, i) => ({
      source: out.chain[i].canonical_id, target: e.canonical_id, rel: "supersedes" }));
    return { nodes, links };
  }
  return null;
};
