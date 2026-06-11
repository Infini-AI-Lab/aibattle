// Single source of truth for the site navbar. Every page includes this script
// (with nav.css); it injects the arena-grouped bar and highlights the active
// item from the current filename. To change the nav anywhere, edit this file.
(function () {
  // current file -> the report href that should be highlighted. Replay pages map
  // to their parent report so the right game stays lit while you watch a replay.
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
    "blackjack_report.html": "blackjack_report.html",
    "leduc_report.html": "leduc_report.html",
    "blotto_report.html": "blotto_report.html",
    "othello_report.html": "othello_report.html"
  };
  var file = location.pathname.split("/").pop() || "index.html";
  var active = ACTIVE[file] || "";

  function a(href, label, cls) {
    var on = href === active ? " active" : "";
    return '<a class="' + cls + on + '" href="' + href + '">' + label + "</a>";
  }

  var html =
    '<a class="brand" href="index.html">🎲 AI Battle Arena</a>' +
    a("index.html", "Overview", "nav") +
    '<a class="navgrp navarena" href="index.html#model">Model Arena</a>' +
    a("connect4_report.html", "🔴 Connect Four", "nav") +
    a("gomoku_report.html", "⚫ Gomoku", "nav") +
    a("kuhn_tournament_report.html", "🃏 Kuhn", "nav") +
    "<span class=\"navclust\">🃏 Hold'em</span>" +
    a("holdem_tournament_report.html", "1-Hand", "nav") +
    a("match_tournament_report.html", "Match", "nav") +
    a("table_tournament_report.html", "Table", "nav") +
    '<span class="navclust">🆕 New games</span>' +
    a("blackjack_report.html", "🃏 Blackjack", "nav") +
    a("leduc_report.html", "🃏 Leduc", "nav") +
    a("blotto_report.html", "⚔️ Blotto", "nav") +
    a("othello_report.html", "⚫ Othello", "nav") +
    '<a class="navgrp navarena" href="index.html#agentic">Agentic Arena<span class="soon">soon</span></a>' +
    '<a class="navgrp" href="gpt_vs_claude/index.html">🥊 GPT vs Claude</a>';

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
