// Navbar for the COACHED arena. Tailored copy of reports/nav.js: same shared
// component, but links resolve within reports/coached/ (every *_report.html here
// is the coached variant) and a back-link returns to the base arena. Replay
// items are omitted (no coached replay viewers are built). To change the base
// nav, edit reports/nav.js instead.
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
    "table_tournament_report.html": "table_tournament_report.html"
  };
  var file = location.pathname.split("/").pop() || "index.html";
  var active = ACTIVE[file] || "";

  function a(href, label, cls) {
    var on = href === active ? " active" : "";
    return '<a class="' + cls + on + '" href="' + href + '">' + label + "</a>";
  }

  var html =
    '<a class="brand" href="index.html">🎓 Coached Arena</a>' +
    a("index.html", "Overview", "nav") +
    '<a class="navgrp navarena" href="index.html#model">Model Arena</a>' +
    a("connect4_report.html", "🔴 Connect Four", "nav") +
    a("gomoku_report.html", "⚫ Gomoku", "nav") +
    a("kuhn_tournament_report.html", "🃏 Kuhn", "nav") +
    "<span class=\"navclust\">🃏 Hold'em</span>" +
    a("holdem_tournament_report.html", "1-Hand", "nav") +
    a("match_tournament_report.html", "Match", "nav") +
    a("table_tournament_report.html", "Table", "nav") +
    '<a class="navgrp" href="../index.html">← Base Arena</a>';

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
