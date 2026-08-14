document$.subscribe(() => {
  if (typeof mermaid !== "undefined") {
    const isDark = document.body.getAttribute("data-md-color-scheme") === "slate";
    mermaid.initialize({
      startOnLoad: false,
      theme: isDark ? "dark" : "default",
      securityLevel: "loose"
    });

    const containers = document.querySelectorAll(".mermaid");
    containers.forEach((el) => {
      const code = el.querySelector("code");
      if (code && !el.getAttribute("data-processed")) {
        el.textContent = code.textContent;
      }
    });

    mermaid.run({
      nodes: document.querySelectorAll(".mermaid:not([data-processed='true'])")
    }).catch((err) => {
      console.warn("Mermaid execution error:", err);
    });
  }
});
