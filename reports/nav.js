// Site nav (markup injected here). TWO bars, one dir-aware file shared by the
// base arena (reports/*.html) and the GPT-vs-Claude pages (reports/gpt_vs_claude/*.html):
// a copy is shipped into that subdir so each page's relative <script src="nav.js">
// loads it.
//   - TOP BAR (horizontal, fixed): brand + site-level pages — Overview (blog),
//     Leaderboard, Featured replays, Q&A, GitHub pinned right.
//   - SIDEBAR (vertical, fixed left): the content tree — Leaderboard + game reports
//     grouped into imperfect-/perfect-info.
// Each link is rewritten with a directory prefix so it resolves from either
// location; replay pages map back to their parent report for highlighting.
(function () {
  var path = location.pathname;
  var inGvc = path.indexOf("/gpt_vs_claude/") !== -1;
  var file = path.split("/").pop() || "index.html";
  var cur = inGvc ? "gvc" : "oss";

  // Prefix to reach each section's pages from wherever we currently are.
  var P = { gvc: inGvc ? "" : "gpt_vs_claude/", oss: inGvc ? "../" : "" };

  // Per-section: which report entry lights up for a given page (a replay page
  // highlights its parent report). gvc and oss share filenames but differ for
  // Hold'em, so the maps are kept separate and matched against the page's dir.
  var ACTIVE = {
    oss: {
      "index.html": "index.html",
      "leaderboard.html": "leaderboard.html",
      "replays.html": "replays.html",
      "qa.html": "qa.html",
      "connect4_report.html": "connect4_report.html",
      "connect4_replay.html": "connect4_report.html",
      "gomoku_report.html": "gomoku_report.html",
      "gomoku_replay.html": "gomoku_report.html",
      "othello_report.html": "othello_report.html",
      "othello_replay.html": "othello_report.html",
      "kuhn_tournament_report.html": "kuhn_tournament_report.html",
      "kuhn_replay.html": "kuhn_tournament_report.html",
      "holdem_tournament_report.html": "holdem_tournament_report.html",
      "holdem_replay.html": "holdem_tournament_report.html",
      "match_tournament_report.html": "match_tournament_report.html",
      "match_replay.html": "match_tournament_report.html",
      "table_tournament_report.html": "table_tournament_report.html",
      "table_replay.html": "table_tournament_report.html",
      "leduc_report.html": "leduc_report.html",
      "leduc_replay.html": "leduc_report.html",
      "blotto_report.html": "blotto_report.html",
      "blotto_replay.html": "blotto_report.html",
      "blackjack_report.html": "blackjack_report.html",
      "blackjack_replay.html": "blackjack_report.html"
    },
    gvc: {
      "index.html": "index.html",
      "connect4_report.html": "connect4_report.html",
      "connect4_replay.html": "connect4_report.html",
      "gomoku_report.html": "gomoku_report.html",
      "gomoku_replay.html": "gomoku_report.html",
      "holdem_1hand_report.html": "holdem_1hand_report.html",
      "holdem_replay.html": "holdem_1hand_report.html",
      "holdem_match_report.html": "holdem_match_report.html",
      "match_replay.html": "holdem_match_report.html"
    }
  };
  var active = (ACTIVE[cur] || {})[file] || "";

  var V = "";
  function a(href, label, cls, section) {
    var on = (section === cur && href === active) ? " active" : "";
    return '<a class="' + cls + on + '" href="' + P[section] + href + V + '">' + label + "</a>";
  }

  var GH_SVG = '<svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>';

  // Horizontal top bar: site-level destinations, GitHub pinned right.
  var topLinks =
    a("index.html", "Overview", "tnav", "oss") +
    a("leaderboard.html", "Leaderboard", "tnav", "oss") +
    a("replays.html", "Featured replays", "tnav", "oss") +
    a("qa.html", "Q&A", "tnav", "oss");
  var ghLink =
    '<a class="tnav gh" href="https://github.com/Infini-AI-Lab/aibattle" target="_blank" rel="noopener" title="GitHub">' +
    GH_SVG + ' GitHub</a>';

  // Vertical sidebar: the content tree — leaderboard + game reports.
  var sideGames =
    '<span class="navclust">Imperfect-info/</span>' +
    a("holdem_tournament_report.html", "Holdem 1hand", "nav navsub", "oss") +
    a("match_tournament_report.html", "Holdem Match", "nav navsub", "oss") +
    a("leduc_report.html", "Leduc Holdem", "nav navsub", "oss") +
    a("kuhn_tournament_report.html", "Kuhn", "nav navsub", "oss") +
    a("blotto_report.html", "Blotto", "nav navsub", "oss") +
    a("blackjack_report.html", "Blackjack", "nav navsub", "oss") +
    '<span class="navclust">Perfect-info/</span>' +
    a("connect4_report.html", "Connect4", "nav navsub", "oss") +
    a("gomoku_report.html", "Gomoku", "nav navsub", "oss");
  var sideLinks =
    a("leaderboard.html", "Leaderboard", "nav navtop", "oss") + sideGames;

  var topbarHtml =
    '<a class="brand" href="' + P.oss + 'index.html' + V + '">🎲 ~/aibattle <span class="prompt">$</span></a>' +
    '<div class="tblinks">' + topLinks + '</div>' +
    '<div class="tbright">' + ghLink + '</div>' +
    '<button type="button" class="navtoggle" aria-label="Menu" aria-expanded="false">☰</button>';

  // Mobile dropdown panel = top links + game tree in one list.
  var menuHtml =
    '<div class="tbmenu">' +
      topLinks.replace(/class="tnav/g, 'class="nav navtop') +
      sideGames +
      ghLink.replace('class="tnav gh"', 'class="nav navtop"') +
    '</div>';

  function mount() {
    // Top bar
    var top = document.querySelector("nav.topbar");
    if (!top) {
      top = document.createElement("nav");
      top.className = "topbar";
      document.body.insertBefore(top, document.body.firstChild);
    }
    top.innerHTML = topbarHtml + menuHtml;
    // Sidebar
    var nav = document.querySelector("nav.navbar");
    if (!nav) {
      nav = document.createElement("nav");
      nav.className = "navbar";
      document.body.insertBefore(nav, top.nextSibling);
    }
    nav.innerHTML = '<div class="navlinks">' + sideLinks + '</div>';
    // Mobile dropdown: ☰ toggles the panel; tapping a link closes it.
    var btn = top.querySelector(".navtoggle");
    if (btn) {
      btn.addEventListener("click", function () {
        var open = top.classList.toggle("open");
        btn.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }
    var ls = top.querySelectorAll(".tbmenu a");
    for (var i = 0; i < ls.length; i++) {
      ls[i].addEventListener("click", function () { top.classList.remove("open"); });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
