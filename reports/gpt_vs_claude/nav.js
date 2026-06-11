// Navbar for the GPT-vs-Claude mini-site. Links resolve within
// reports/gpt_vs_claude/; a back-link returns to the base arena.
(function () {
  var ACTIVE = {
    "index.html": "index.html",
    "connect4_report.html": "connect4_report.html",
    "connect4_replay.html": "connect4_report.html",
    "gomoku_report.html": "gomoku_report.html",
    "gomoku_replay.html": "gomoku_report.html",
    "holdem_1hand_report.html": "holdem_1hand_report.html",
    "holdem_replay.html": "holdem_1hand_report.html",
    "holdem_match_report.html": "holdem_match_report.html",
    "match_replay.html": "holdem_match_report.html"
  };
  var file = location.pathname.split("/").pop() || "index.html";
  var active = ACTIVE[file] || "";
  function a(href, label, cls) {
    var on = href === active ? " active" : "";
    return '<a class="' + cls + on + '" href="' + href + '">' + label + "</a>";
  }
  var html =
    '<a class="brand" href="index.html">🥊 GPT vs Claude</a>' +
    a("index.html", "Overview", "nav") +
    a("connect4_report.html", "🔴 Connect Four", "nav") +
    a("gomoku_report.html", "⚫ Gomoku", "nav") +
    "<span class=\"navclust\">🃏 Hold'em</span>" +
    a("holdem_1hand_report.html", "1-Hand", "nav") +
    a("holdem_match_report.html", "Match", "nav") +
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
  } else { mount(); }
})();
