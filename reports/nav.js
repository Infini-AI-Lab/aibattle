// Site navbar (left sidebar). ONE dir-aware file shared by the base arena
// (reports/*.html) and the GPT-vs-Claude pages (reports/gpt_vs_claude/*.html):
// a copy is shipped into that subdir so each page's relative <script src="nav.js">
// loads it. The main arena lists a single set of games split into perfect-/
// imperfect-info groups (the gvc subpages keep their own section logic). Each
// link is rewritten with a directory prefix so it resolves from either
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

  // V busts the browser's heuristic cache of the page HTML (the dev server
  // sends no Cache-Control). Bump it when the nav or pages are restyled.
  var V = "?v=31";
  function a(href, label, cls, section) {
    var on = (section === cur && href === active) ? " active" : "";
    return '<a class="' + cls + on + '" href="' + P[section] + href + V + '">' + label + "</a>";
  }

  var html =
    '<a class="brand" href="' + P.oss + 'index.html' + V + '">🎲 ~/aibattle <span class="prompt">$</span></a>' +
    a("index.html", "Overview", "nav navtop", "oss") +

    // Single arena now — imperfect- and perfect-info game groups, no top label.
    '<span class="navclust">Imperfect-info/</span>' +
    a("holdem_tournament_report.html", "Holdem 1hand", "nav navsub", "oss") +
    a("match_tournament_report.html", "Holdem Match", "nav navsub", "oss") +
    a("leduc_report.html", "Leduc Holdem", "nav navsub", "oss") +
    a("kuhn_tournament_report.html", "Kuhn", "nav navsub", "oss") +
    a("blotto_report.html", "Blotto", "nav navsub", "oss") +
    a("blackjack_report.html", "Blackjack", "nav navsub", "oss") +
    '<span class="navclust">Perfect-info/</span>' +
    a("connect4_report.html", "Connect4", "nav navsub", "oss") +
    a("gomoku_report.html", "Gomoku", "nav navsub", "oss") +
    a("qa.html", "Q&A", "nav navtop", "oss") +

    // Compact mobile bar: shown only under 760px (CSS hides the list above and
    // this below). Four columns; the two category links jump to the matching
    // group on the Overview page (anchors #imperfect / #perfect there).
    '<div class="navmobile">' +
    a("index.html", "Overview", "navm", "oss") +
    '<a class="navm" href="' + P.oss + 'index.html' + V + '#imperfect">Imperfect-info</a>' +
    '<a class="navm" href="' + P.oss + 'index.html' + V + '#perfect">Perfect-info</a>' +
    a("qa.html", "Q&A", "navm", "oss") +
    '</div>';

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
