// The Kino.JS entrypoint of Transmuter.Graph.as_graph/2. The data gives the scripts, from
// the assets of the kino, so the view works without a network connection.
export async function init(ctx, data) {
  ctx.importCSS("viewer.css");
  for (const script of data.scripts) await ctx.importJS(script);
  ctx.root.style.height = "640px";
  window.TransmuterGraphView.render(ctx.root, data);
}
