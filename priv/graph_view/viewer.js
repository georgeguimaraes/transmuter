// The view of Transmuter.Graph. Elixir builds the tree of groups (Transmuter.Graph.tree/2),
// and this script only draws it and opens and closes groups. It needs cytoscape, dagre and
// cytoscape-dagre, loaded before it.
(function () {
  const SEGMENT_COLORS = ["#dbeafe", "#dcfce7", "#fef3c7", "#f3e8ff", "#cffafe", "#ffe4e6"];
  const NO_SEGMENT = "#e5e7eb";
  const FAN_OUT = 4;

  const STYLE = [
    { selector: "node", style: { "shape": "round-rectangle", "background-color": "data(color)", "border-width": 1, "border-color": "#64748b", "label": "data(label)", "text-wrap": "wrap", "text-valign": "center", "font-size": 10, "width": "label", "height": "label", "padding": 8 } },
    { selector: "node.group", style: { "background-fill": "linear-gradient", "background-gradient-direction": "to-right", "background-gradient-stop-colors": "data(stops)", "border-width": 2, "font-size": 11, "padding": 10 } },
    { selector: "node.static", style: { "border-style": "dashed", "color": "#6b7280" } },
    { selector: "node.fused", style: { "border-width": 3 } },
    { selector: "node.break", style: { "border-color": "#b91c1c", "border-width": 3, "color": "#991b1b" } },
    { selector: "node.break[kind = 'node']", style: { "background-color": "#fecaca" } },
    { selector: "node[kind = 'container']", style: { "background-color": "#fff", "background-opacity": 0.6, "border-color": "#94a3b8", "label": "data(label)", "text-valign": "top", "text-halign": "center", "font-size": 11, "color": "#475569" } },
    { selector: "edge", style: { "width": 1, "line-color": "#94a3b8", "target-arrow-color": "#94a3b8", "target-arrow-shape": "triangle", "curve-style": "bezier", "arrow-scale": 0.7 } },
    { selector: "edge.fan", style: { "opacity": 0.12 } },
    { selector: "node:selected", style: { "overlay-opacity": 0.15, "overlay-color": "#2563eb" } },
    { selector: "edge.lit", style: { "opacity": 1, "line-color": "#2563eb", "target-arrow-color": "#2563eb", "width": 2 } },
  ];

  const escape = text => String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const segmentColor = segment => SEGMENT_COLORS[segment % SEGMENT_COLORS.length];
  const shapeText = shape => shape ? "[" + shape.shape.join(", ") + "]" + (shape.type ? " " + shape.type : "") : "";

  function groupCounts(counts) {
    return [
      counts.nodes.toLocaleString("en") + (counts.nodes === 1 ? " op" : " ops"),
      counts.static && counts.static.toLocaleString("en") + " static",
      counts.fused && counts.fused + " fused",
      counts.breaks && counts.breaks + (counts.breaks === 1 ? " break" : " breaks"),
    ].filter(Boolean).join(" · ");
  }

  function render(root, data) {
    if (!cytoscape.__transmuterDagre) {
      cytoscape.use(cytoscapeDagre);
      cytoscape.__transmuterDagre = true;
    }

    root.innerHTML = `<div class="tgv">
      <header>
        <h1>${escape(data.title)}</h1>
        <span>${data.tree.counts.nodes.toLocaleString("en")} ops${data.segments.length ? " · " + data.segments.length + " segments" : ""}</span>
        <button data-action="collapse">Collapse all</button>
        <button data-action="fit">Fit</button>
        <span class="legend"><span>op</span><span class="static">static</span><span class="fused">fused</span><span class="break">break</span></span>
      </header>
      <div class="tgv-cy"></div>
      <aside></aside>
    </div>`;

    const info = root.querySelector("aside");
    const byId = new Map();
    const parent = new Map();

    (function walk(group) {
      byId.set(group.id, group);
      for (const child of group.children) {
        parent.set(child.id, group.id);
        if (child.kind === "group") walk(child);
        else byId.set(child.id, child);
      }
    })(data.tree);

    const leaves = [...byId.values()].filter(e => e.kind === "node");
    let expanded = new Set();

    // The shown element of a node or a group: its outermost closed group, or itself.
    function shown(id) {
      const chain = [];
      for (let p = id; p !== data.tree.id; p = parent.get(p)) chain.unshift(p);
      for (const ancestor of chain) {
        if (byId.get(ancestor).kind === "group" && !expanded.has(ancestor)) return ancestor;
      }
      return id;
    }

    function groupElement(group, container) {
      const colors = group.segments.length ? group.segments.map(segmentColor) : [NO_SEGMENT];
      return {
        data: {
          id: group.id, parent: container, kind: "group",
          label: group.name + "\n" + groupCounts(group.counts),
          color: colors[0],
          stops: (colors.length > 1 ? colors : [colors[0], colors[0]]).join(" "),
        },
        classes: "group" + (group.counts.breaks ? " break" : ""),
      };
    }

    function leafElement(node, container) {
      const second = shapeText(node.shapes && node.shapes[0]) || node.label || "";
      return {
        data: {
          id: node.id, parent: container, kind: "node",
          label: node.op_type + (second ? "\n" + second : ""),
          color: node.segment === null ? NO_SEGMENT : segmentColor(node.segment),
        },
        classes: [node.static && "static", node.fused.length && "fused", node.break && "break"].filter(Boolean).join(" "),
      };
    }

    function elements() {
      const result = [];

      (function add(group, container) {
        for (const child of group.children) {
          if (child.kind === "node") result.push(leafElement(child, container));
          else if (expanded.has(child.id)) {
            result.push({ data: { id: child.id, parent: container, kind: "container", label: child.scope } });
            add(child, child.id);
          } else result.push(groupElement(child, container));
        }
      })(data.tree, undefined);

      const edges = new Map();
      for (const leaf of leaves) {
        const to = shown(leaf.id);
        for (const read of leaf.reads) {
          if (!byId.has(read)) continue;
          const from = shown(read);
          if (from !== to) edges.set(from + "->" + to, { from, to });
        }
      }

      const fanOut = new Map();
      for (const { from } of edges.values()) fanOut.set(from, (fanOut.get(from) || 0) + 1);
      for (const [id, { from, to }] of edges) {
        result.push({ data: { id, source: from, target: to }, classes: fanOut.get(from) > FAN_OUT ? "fan" : "" });
      }

      return result;
    }

    const cy = cytoscape({ container: root.querySelector(".tgv-cy"), wheelSensitivity: 0.3, style: STYLE });

    function draw(focus) {
      cy.elements().remove();
      cy.add(elements());
      cy.layout({ name: "dagre", rankDir: "TB", nodeSep: 20, rankSep: 40, animate: false }).run();
      const target = focus && cy.getElementById(focus);
      if (target && target.nonempty()) cy.fit(target, 40);
      else cy.fit(undefined, 20);
    }

    function segmentText(node) {
      if (node.break) return `between segment ${node.break.after} and ${node.break.after + 1}`;
      if (!data.segments.length) return "unknown: the view is static only";
      return node.segment === null ? "none (the node did not run)" : String(node.segment);
    }

    function nodeDetails(node) {
      const outputs = node.outputs.map((name, i) => {
        const shape = shapeText(node.shapes && node.shapes[i]);
        return escape(name) + (shape ? ` <b>${escape(shape)}</b>` : "");
      });

      const attributes = Object.entries(node.attributes).map(([name, value]) =>
        escape(name) + " = " + escape(typeof value === "string" ? value : JSON.stringify(value)));

      info.innerHTML = `<h2>${escape(node.op_type)}</h2><dl>
        <dt>Name</dt><dd>${escape(node.name || "")}</dd>
        <dt>Placement</dt><dd>${escape(node.placement ?? "unknown")}</dd>
        <dt>Segment</dt><dd>${escape(segmentText(node))}</dd>
        <dt>Static outputs</dt><dd>${escape(node.static ?? "unknown")}</dd>
        ${node.break ? `<dt>Graph break</dt><dd>${escape(node.break.reason)}</dd>` : ""}
        ${node.literals.length ? `<dt>Literal inputs</dt><dd>${node.literals.map(escape).join("<br>")}</dd>` : ""}
        ${node.fused.length ? `<dt>Fused from ${node.fused.length} nodes</dt><dd>${node.fused.map(escape).join("<br>")}</dd>` : ""}
        ${attributes.length ? `<dt>Attributes</dt><dd>${attributes.join("<br>")}</dd>` : ""}
        <dt>Inputs</dt><dd>${node.inputs.map(escape).join("<br>")}</dd>
        <dt>Outputs</dt><dd>${outputs.join("<br>")}</dd></dl>`;
    }

    function groupDetails(group) {
      const constants = group.counts.constants ? ` · ${group.counts.constants} Constant nodes hidden` : "";
      info.innerHTML = `<h2>${escape(group.scope)}</h2><p>${escape(groupCounts(group.counts) + constants)}</p>
        ${group.segments.length ? `<p>Segments ${group.segments.join(", ")}</p>` : ""}
        <p class="hint">Click the group to open it.</p>`;
    }

    function overview() {
      const segments = data.segments.map(s =>
        `<dd>${s.index}: ${s.compiled ? "compiled" : "eager"}, stops at ${escape(s.stop)}${s.reason ? " (" + escape(s.reason) + ")" : ""}</dd>`);
      info.innerHTML = `<p class="hint">Click a group to open it. Click the frame of an open group to close it. Click a node to see its details here. Scroll to zoom, drag to pan.</p>
        ${segments.length ? `<dl><dt>Segments</dt>${segments.join("")}</dl>` : `<p class="hint">No segments: the view is static only. Give <code>inputs:</code> to see the segments.</p>`}`;
    }

    cy.on("tap", "node", e => {
      const id = e.target.id();
      const kind = e.target.data("kind");
      cy.edges().removeClass("lit");
      e.target.connectedEdges().addClass("lit");

      if (kind === "group") {
        expanded.add(id);
        draw(id);
      } else if (kind === "container") {
        for (const open of [...expanded]) if (open === id || open.startsWith(id + "/")) expanded.delete(open);
        draw();
      } else nodeDetails(byId.get(id));
    });

    cy.on("mouseover", "node[kind = 'group']", e => groupDetails(byId.get(e.target.id())));

    root.querySelector('[data-action="collapse"]').onclick = () => { expanded = new Set(); draw(); };
    root.querySelector('[data-action="fit"]').onclick = () => cy.fit(undefined, 20);

    overview();
    draw();
  }

  window.TransmuterGraphView = { render };
})();
