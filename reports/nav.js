// Site navbar (left sidebar). Links resolve within reports/. Replay viewers
// aren't listed here — each report links to its own replay page, and ACTIVE
// maps replay pages back to their parent report so the right item stays lit.
(function () {
  var ACTIVE = {
    "index.html": "index.html",
    "connect4_report.html": "connect4_report.html",
    "connect4_replay.html": "connect4_report.html",
    "gomoku_report.html": "gomoku_report.html",
    "gomoku_replay.html": "gomoku_report.html",
    "kuhn_tournament_report.html": "kuhn_tournament_report.html",
    "holdem_tournament_report.html": "holdem_tournament_report.html",
    "holdem_replay.html": "holdem_tournament_report.html",
    "match_tournament_report.html": "match_tournament_report.html",
    "match_replay.html": "match_tournament_report.html",
    "table_tournament_report.html": "table_tournament_report.html",
    "table_replay.html": "table_tournament_report.html"
  };
  var file = location.pathname.split("/").pop() || "index.html";
  var active = ACTIVE[file] || "";

  // V busts the browser's heuristic cache of the page HTML (the dev server
  // sends no Cache-Control). Bump it when pages are restyled. ACTIVE matching
  // uses the bare filename, so the query string never affects highlighting.
  var V = "?v=7";
  function a(href, label, cls) {
    var on = href === active ? " active" : "";
    return '<a class="' + cls + on + '" href="' + href + V + '">' + label + "</a>";
  }

  var html =
    '<a class="brand" href="index.html' + V + '">🎲 ~/aibattle <span class="prompt">$</span></a>' +
    a("index.html", "overview", "nav") +
    a("connect4_report.html", "connect4", "nav") +
    a("gomoku_report.html", "gomoku", "nav") +
    a("kuhn_tournament_report.html", "kuhn", "nav") +
    '<span class="navclust">holdem/</span>' +
    a("holdem_tournament_report.html", "1hand", "nav navsub") +
    a("match_tournament_report.html", "match", "nav navsub") +
    a("table_tournament_report.html", "table", "nav navsub");

  function mount() {
    var nav = document.querySelector("nav.navbar");
    if (!nav) {
      nav = document.createElement("nav");
      nav.className = "navbar";
      document.body.insertBefore(nav, document.body.firstChild);
    }
    nav.innerHTML = html;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
