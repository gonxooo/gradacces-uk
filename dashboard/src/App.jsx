import { useState, useEffect, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend, CartesianGrid
} from "recharts";

/*
  Dashboard.jsx - GradAccess UK
  ─────────────────────────────────────────────────────────────────
  Final Year Project: "Are 'Entry-Level' Jobs Really Entry-Level?"
  An Empirical Analysis of UK Graduate Software Engineering Postings

  Two tabs:
    - Overview: aggregate stats for the 151-posting dataset
    - Live Analyser: paste any posting and compute GAS in real time
      using an exact JavaScript port of the Python pipeline.
*/

const CAT_COLORS = {
  bootcamp:          "#f97316",
  genuine_junior:    "#10b981",
  paradox:           "#ef4444",
  ambiguous:         "#3b82f6",
  off_topic:         "#6b7280",
};

const CAT_LABELS = {
  bootcamp:          "Bootcamp (training for sale)",
  genuine_junior:    "Genuine Junior",
  paradox:           "Graduate Paradox",
  ambiguous:         "Ambiguous",
  off_topic:         "Off-topic",
};

const CAT_DESCRIPTIONS = {
  bootcamp:          "Paid training programmes indexed by the job platform as employment offers.",
  genuine_junior:    "Roles explicitly welcoming recent graduates or requiring no prior experience.",
  paradox:           "Roles labelled 'junior' yet demanding ≥2 years of prior experience.",
  ambiguous:         "Roles without explicit experience signals either way.",
  off_topic:         "Roles that slipped through title filtering but are not software engineering.",
};


// ═══════════════════════════════════════════════════════════════════
// CLASSIFIER LOGIC — ported from the Python pipeline (nlp_pipeline.py
// and scoring.py). Any change in the Python code must be mirrored here
// to keep the live analyser consistent with the aggregate analysis.
// ═══════════════════════════════════════════════════════════════════

const TECH_DICTIONARY = {
  javascript:   [/\bjavascript\b/i],
  typescript:   [/\btypescript\b/i],
  python:       [/\bpython\b/i],
  java:         [/\bjava\b(?!script)/i],
  "c#":         [/c#/i, /\.net/i, /\bcsharp\b/i],
  "c++":        [/c\+\+/i, /\bcpp\b/i],
  go:           [/\bgolang\b/i],
  rust:         [/\brust\b/i],
  php:          [/\bphp\b/i],
  ruby:         [/\bruby\b/i],
  kotlin:       [/\bkotlin\b/i],
  swift:        [/\bswift\b/i],
  scala:        [/\bscala\b/i],
  dart:         [/\bdart\b/i],
  sql:          [/\bsql\b/i],
  react:        [/\breact\.?js\b/i, /\breact\b/i],
  vue:          [/\bvue\.?js\b/i, /\bvue\b/i],
  angular:      [/\bangular\b/i],
  svelte:       [/\bsvelte\b/i],
  nextjs:       [/\bnext\.?js\b/i],
  html:         [/\bhtml5?\b/i],
  css:          [/\bcss3?\b/i, /\bsass\b/i, /\btailwind\b/i],
  nodejs:       [/\bnode\.?js\b/i, /\bnodejs\b/i],
  express:      [/\bexpress\.?js\b/i],
  spring:       [/\bspring\s+boot\b/i, /\bspring\b/i],
  django:       [/\bdjango\b/i],
  flask:        [/\bflask\b/i],
  laravel:      [/\blaravel\b/i],
  android:      [/\bandroid\b/i],
  ios:          [/\bios\b/i],
  flutter:      [/\bflutter\b/i],
  postgresql:   [/\bpostgres(?:ql)?\b/i],
  mysql:        [/\bmysql\b/i],
  mongodb:      [/\bmongo(?:db)?\b/i],
  redis:        [/\bredis\b/i],
  aws:          [/\baws\b/i, /amazon\s+web\s+services/i],
  azure:        [/\bazure\b/i],
  gcp:          [/\bgcp\b/i, /google\s+cloud/i],
  docker:       [/\bdocker\b/i],
  kubernetes:   [/\bkubernetes\b/i, /\bk8s\b/i],
  git:          [/\bgit(?:hub|lab)?\b/i],
  linux:        [/\blinux\b/i, /\bunix\b/i],
  rest:         [/\brest(?:ful)?\s+api\b/i],
  graphql:      [/\bgraphql\b/i],
  agile:        [/\bagile\b/i, /\bscrum\b/i],
  "ci/cd":      [/ci\/cd/i, /continuous\s+integration/i],
};

const BOOTCAMP_PATTERNS = [
  /skills\s+shortages\s+in\s+the\s+it\s+sector/i,
  /industry\s+recognised\s+certifications/i,
  /career\s+seekers\s+and\s+career\s+changers/i,
  /invest\s+(?:some\s+)?time\s+and\s+money/i,
  /training\s+and\s+placement\s+programme/i,
  /skills\s+bootcamp/i,
  /govt[- ]funded/i,
  /government[- ]funded\s+(?:skills\s+)?bootcamp/i,
  /fully[- ]funded\s+(?:cyber\s+security\s+)?course/i,
  /ncfe\s+certificate/i,
  /training\s+programme\s+(?:is\s+)?made\s+for\s+you/i,
  /build\s+the\s+foundations\s+for\s+a\s+new\s+career/i,
  /help\s+place\s+graduates\s+from\s+this\s+programme/i,
];

const YEARS_PATTERNS = [
  /seniority\s*:?\s*[^.]*?\(\s*(\d+)\s*[-\u2013]\s*\d+\s*years?/i,
  /(\d+)\s*[-\u2013]\s*\d+\s*years?\b[^.]{0,100}?(?:experience|commercial|industry|professional|working|software\s+development)/i,
  /(\d+)\s*[-\u2013]\s*\d+\s*years?\b\s*(?:commercial|industry|professional|relevant)/i,
  /(\d+)\s*\+?\s*years?\b[^.]{0,60}?(?:experience|commercial|industry|professional|working|hands[- ]on|software)/i,
  /(?:minimum|at\s+least|minimum\s+of)\s+(\d+)\s*\+?\s*years?\b/i,
  /(?:experience|commercial)\s+(?:of|with|in|using)\s+(?:at\s+least\s+)?(\d+)\s*\+?\s*years?\b/i,
  /(\d+)\s*\+?\s*years?\s+(?:of\s+)?pqe\b/i,
];

const SPELLED_NUMBERS = { one: 1, two: 2, three: 3, four: 4, five: 5 };
const SPELLED_PATTERN = /\b(one|two|three|four|five)\s+years?\b[^.]{0,40}?(?:experience|commercial)/i;

const NO_EXP_PATTERNS = [
  /no\s+(?:prior\s+)?(?:commercial\s+)?experience\s+(?:required|needed|necessary|essential)/i,
  /no\s+previous\s+experience/i,
  /fresh\s+graduate/i,
  /recent\s+graduate/i,
  /new\s+graduate/i,
  /training\s+(?:will\s+be\s+)?provided/i,
  /full\s+training/i,
  /we\s+will\s+train/i,
  /structured\s+(?:training|programme|program)/i,
  /early[- ]career/i,
  /from\s+university/i,
  /supportive\s+environment/i,
  /entry[- ]level\s+position/i,
  /career\s+(?:journey|path|starter|changer)/i,
  /kick[- ]?start\s+your\s+career/i,
  /alongside\s+(?:senior|experienced)\s+engineers?/i,
  /mentorship/i,
  /learn\s+(?:on\s+the\s+job|and\s+grow)/i,
];

const DEGREE_PATTERNS = {
  phd:      /\b(?:ph\.?\s*d|doctorate|doctoral)\b/i,
  master:   /\b(?:master(?:['\u2019]s)?|msc|m\.sc\b|mba)\b/i,
  bachelor: /\b(?:bachelor(?:['\u2019]s)?|bsc|b\.sc\b|beng|honours\s+degree|degree\s+in|degree\s+level|university\s+degree|relevant\s+degree)\b/i,
};

const OFF_TOPIC_PATTERNS = [
  /customer\s+service/i,
  /account(?:s|ing|ant|ancy)\b/i,
  /\bsales\b/i,
  /\bteacher\b/i,
  /recruit(?:er|ment)/i,
  /\bmarketing\b/i,
  /\bhr\b/i,
  /human\s+resources/i,
];

const SOFTWARE_PATTERNS = [
  /software\s+(?:engineer|developer|development)/i,
  /\bdeveloper\b/i,
  /programm(?:er|ing)/i,
  /full[- ]stack|backend|front[- ]end/i,
  /\bcoding\b/i,
];

const EXCLUSIONARY_PATTERNS = [
  /hit\s+the\s+ground\s+running/i,
  /proven\s+track\s+record/i,
  /extensive\s+experience/i,
  /significant\s+experience/i,
  /deep\s+knowledge/i,
  /expert(?:ise)?\s+in/i,
  /mastery\s+of/i,
  /\barchitect\b/i,
];


function analyseposting(title, description) {
  const t = title || "";
  const d = description || "";
  const combined = `${t} . ${d}`;

  // Tech extraction
  const technologies = [];
  for (const [name, patterns] of Object.entries(TECH_DICTIONARY)) {
    if (patterns.some(p => p.test(combined))) technologies.push(name);
  }

  // Years of experience
  let years = null;
  const foundYears = [];
  for (const pat of YEARS_PATTERNS) {
    const matches = combined.matchAll(new RegExp(pat.source, pat.flags + "g"));
    for (const m of matches) {
      const n = parseInt(m[1], 10);
      if (!isNaN(n) && n >= 0 && n <= 20) foundYears.push(n);
    }
  }
  const spelled = SPELLED_PATTERN.exec(combined);
  if (spelled && SPELLED_NUMBERS[spelled[1].toLowerCase()]) {
    foundYears.push(SPELLED_NUMBERS[spelled[1].toLowerCase()]);
  }
  if (foundYears.length) years = Math.min(...foundYears);

  // Other signals
  const noExpSignal = NO_EXP_PATTERNS.some(p => p.test(combined));
  const isBootcamp = BOOTCAMP_PATTERNS.filter(p => p.test(combined)).length >= 2;
  const bootcampMatches = BOOTCAMP_PATTERNS.filter(p => p.test(combined));

  const offTopicHits = OFF_TOPIC_PATTERNS.filter(p => p.test(combined));
  const softwareHits = SOFTWARE_PATTERNS.filter(p => p.test(combined));
  const isOffTopic = offTopicHits.length > 0 && softwareHits.length === 0;

  let degree = null;
  if (DEGREE_PATTERNS.phd.test(combined))      degree = "phd";
  else if (DEGREE_PATTERNS.master.test(combined))   degree = "master";
  else if (DEGREE_PATTERNS.bachelor.test(combined)) degree = "bachelor";

  // Detected patterns (for UI explanation)
  const detectedInclusive = NO_EXP_PATTERNS.filter(p => p.test(combined));
  const detectedExclusive = EXCLUSIONARY_PATTERNS.filter(p => p.test(combined));

  // Category
  let category;
  if (isOffTopic) {
    category = "off_topic";
  } else if (isBootcamp) {
    category = "bootcamp";
  } else if (years !== null && years >= 2) {
    category = "paradox";
  } else if ((years !== null && years <= 1) || noExpSignal) {
    category = "genuine_junior";
  } else {
    category = "ambiguous";
  }

  // GAS computation (mirror of scoring.py)
  let components = null;
  let gas = null;

  if (category === "off_topic") {
    gas = 0;
  } else if (category === "bootcamp") {
    gas = 15;
  } else {
    const sExp = (() => {
      if (noExpSignal) return 100;
      if (years === null) return 70;
      if (years <= 0) return 100;
      if (years === 1) return 70;
      if (years === 2) return 35;
      if (years === 3) return 10;
      return 0;
    })();

    const sDegree = (() => {
      if (degree === null) return 100;
      if (degree === "bachelor") return 100;
      if (degree === "master") return 40;
      if (degree === "phd") return 0;
      return 80;
    })();

    const techCount = technologies.length;
    const sTech = (() => {
      if (techCount <= 3) return 100;
      if (techCount <= 5) return 85;
      if (techCount <= 7) return 65;
      if (techCount <= 10) return 40;
      return 15;
    })();

    let sLang = 70;
    for (const p of NO_EXP_PATTERNS) if (p.test(combined)) sLang += 8;
    for (const p of EXCLUSIONARY_PATTERNS) if (p.test(combined)) sLang -= 10;
    sLang = Math.max(0, Math.min(100, sLang));

    const sSalary = 60; // we don't know salary when user pastes text — neutral

    components = {
      experience: sExp,
      degree: sDegree,
      tech_breadth: sTech,
      language: sLang,
      salary_transparency: sSalary,
    };
    gas = Math.round(
      (sExp * 0.35 + sDegree * 0.20 + sTech * 0.20 +
       sLang * 0.15 + sSalary * 0.10) * 10
    ) / 10;
  }

  return {
    category, gas, components,
    technologies, years, noExpSignal, degree,
    isBootcamp, isOffTopic,
    bootcampMatches: bootcampMatches.map(p => p.source),
    detectedInclusive: detectedInclusive.map(p => p.source),
    detectedExclusive: detectedExclusive.map(p => p.source),
  };
}


// ═══════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════

export default function App() {
  const [tab, setTab] = useState("overview");
  const [stats, setStats] = useState(null);
  const [postings, setPostings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    Promise.all([
      fetch("/aggregate_stats.json").then(r => r.ok ? r.json() : Promise.reject("stats")),
      fetch("/scored.json").then(r => r.ok ? r.json() : Promise.reject("scored")),
    ])
      .then(([s, p]) => { setStats(s); setPostings(p); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, []);

  const filtered = useMemo(() => {
    return postings.filter(p => {
      if (categoryFilter !== "all" && p.category !== categoryFilter) return false;
      if (searchTerm) {
        const q = searchTerm.toLowerCase();
        const h = `${p.title} ${p.company ?? ""} ${p.location_display ?? ""}`.toLowerCase();
        if (!h.includes(q)) return false;
      }
      return true;
    });
  }, [postings, categoryFilter, searchTerm]);

  if (loading) return <LoadingScreen />;
  if (error)   return <ErrorScreen message={error} />;

  return (
    <div style={pageStyle}>
      <Header tab={tab} setTab={setTab} />

      <div style={containerStyle}>
        {tab === "overview" && (
          <OverviewTab
            stats={stats}
            postings={postings}
            filtered={filtered}
            categoryFilter={categoryFilter}
            setCategoryFilter={setCategoryFilter}
            searchTerm={searchTerm}
            setSearchTerm={setSearchTerm}
          />
        )}
        {tab === "analyser" && <AnalyserTab />}
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════════
// OVERVIEW TAB
// ═══════════════════════════════════════════════════════════════════

function OverviewTab({ stats, postings, filtered, categoryFilter, setCategoryFilter, searchTerm, setSearchTerm }) {
  const categoryData = Object.entries(stats.category_distribution || {})
    .map(([cat, count]) => ({
      name: CAT_LABELS[cat] || cat,
      value: count, cat,
      pct: (count / stats.total_postings * 100).toFixed(1),
    }));

  const yearsData = Object.entries(stats.years_experience_distribution || {})
    .map(([years, count]) => ({ years: `${years} yr`, count: Number(count) }))
    .sort((a, b) => parseInt(a.years) - parseInt(b.years));

  const techData = (stats.top_20_technologies || [])
    .slice(0, 12)
    .map(([name, count]) => ({ name, count }));

  const regionData = Object.entries(stats.region_avg_gas || {})
    .map(([r, gas]) => ({ region: r, gas }))
    .sort((a, b) => b.gas - a.gas)
    .slice(0, 10);

  return (
    <>
      <Intro stats={stats} />
      <HeadlineNumbers stats={stats} />

      <Section title="Category breakdown (the core finding)">
        <div style={gridTwo}>
          <Card>
            <CardTitle>Postings classified into 5 categories</CardTitle>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={categoryData} dataKey="value" nameKey="name" outerRadius={100}
                     label={e => `${e.pct}%`}>
                  {categoryData.map((d, i) => <Cell key={i} fill={CAT_COLORS[d.cat]} />)}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <CardTitle>What each category means</CardTitle>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 8 }}>
              {Object.keys(CAT_LABELS).map(cat => {
                const count = stats.category_distribution?.[cat] ?? 0;
                const pct = (count / stats.total_postings * 100).toFixed(1);
                return (
                  <div key={cat} style={{
                    padding: "10px 12px",
                    borderLeft: `4px solid ${CAT_COLORS[cat]}`,
                    background: "#0f172a", borderRadius: 6,
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                      <span style={{ color: CAT_COLORS[cat], fontSize: 13, fontWeight: 700 }}>
                        {CAT_LABELS[cat]}
                      </span>
                      <span style={{ color: "#cbd5e1", fontSize: 13, fontFamily: "monospace" }}>
                        {count} &nbsp; ({pct}%)
                      </span>
                    </div>
                    <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 4, lineHeight: 1.5 }}>
                      {CAT_DESCRIPTIONS[cat]}
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      </Section>

      <Section title="RQ1 - Distribution of stated experience requirements">
        <Card>
          <CardTitle>Years of commercial experience demanded (where explicitly stated)</CardTitle>
          {yearsData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={yearsData}>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis dataKey="years" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {yearsData.map((d, i) => {
                    const n = parseInt(d.years);
                    const color = n <= 1 ? "#10b981" : n === 2 ? "#f59e0b" : "#ef4444";
                    return <Cell key={i} fill={color} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (<div style={{ padding: 40, textAlign: "center", color: "#64748b" }}>No data</div>)}
          <div style={explainStyle}>
            Green bars (0-1 years) are accessible to recent graduates. Amber (2 years) represent
            a borderline case. Red bars (3+ years) represent the graduate paradox.
          </div>
        </Card>
      </Section>

      <Section title="Top technologies demanded (real roles only)">
        <Card>
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={techData} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis type="number" stroke="#94a3b8" fontSize={12} />
              <YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={12} width={75} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" fill="#8b5cf6" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </Section>

      {regionData.length > 0 && (
        <Section title="RQ2 - Regional variation">
          <Card>
            <CardTitle>Average GAS by UK region (regions with at least 3 postings)</CardTitle>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={regionData}>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis dataKey="region" stroke="#94a3b8" fontSize={11}
                       interval={0} angle={-25} textAnchor="end" height={70} />
                <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="gas" fill="#10b981" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Section>
      )}

      <Section title="Explore the dataset">
        <Card>
          <div style={filtersRow}>
            <input type="text" value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
                   placeholder="Search title, company, or location..." style={searchInputStyle} />
            <select value={categoryFilter} onChange={e => setCategoryFilter(e.target.value)} style={selectStyle}>
              <option value="all">All categories</option>
              {Object.entries(CAT_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <span style={{ color: "#64748b", fontSize: 12, marginLeft: "auto" }}>
              {filtered.length} of {postings.length} postings
            </span>
          </div>

          <div style={tableWrap}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>GAS</th>
                  <th style={thStyle}>Category</th>
                  <th style={thStyle}>Title</th>
                  <th style={thStyle}>Company</th>
                  <th style={thStyle}>Location</th>
                  <th style={thStyle}>Exp.</th>
                  <th style={thStyle}>Salary</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 80).map(p => (
                  <tr key={p.id} style={{ borderTop: "1px solid #1e293b" }}>
                    <td style={{ ...tdStyle, fontFamily: "monospace", fontWeight: 700,
                                  color: p._scoring?.gas >= 75 ? "#10b981"
                                       : p._scoring?.gas >= 50 ? "#f59e0b" : "#ef4444" }}>
                      {p._scoring?.gas ?? "—"}
                    </td>
                    <td style={tdStyle}>
                      <span style={{
                        background: CAT_COLORS[p.category] + "25",
                        color:      CAT_COLORS[p.category],
                        padding:    "3px 8px", borderRadius: 5,
                        fontSize: 11, fontWeight: 600,
                      }}>{CAT_LABELS[p.category]}</span>
                    </td>
                    <td style={tdStyle}>
                      {p.adzuna_url ? (
                        <a href={p.adzuna_url} target="_blank" rel="noreferrer" style={linkStyle}>{p.title}</a>
                      ) : p.title}
                    </td>
                    <td style={tdStyle}>{p.company || "—"}</td>
                    <td style={tdStyle}>{p.location_display || "—"}</td>
                    <td style={tdStyle}>{p.years_experience ?? "—"}</td>
                    <td style={tdStyle}>
                      {p.salary_min ? `£${Math.round(p.salary_min/1000)}k` : "—"}
                      {p.salary_max && p.salary_max !== p.salary_min
                        ? `-${Math.round(p.salary_max/1000)}k` : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filtered.length > 80 && (
              <div style={{ padding: 12, textAlign: "center", color: "#64748b", fontSize: 11 }}>
                Showing first 80 of {filtered.length} - refine filters to narrow down.
              </div>
            )}
          </div>
        </Card>
      </Section>

      <Methodology />
      <Footer />
    </>
  );
}


// ═══════════════════════════════════════════════════════════════════
// LIVE ANALYSER TAB
// ═══════════════════════════════════════════════════════════════════

const EXAMPLE_POSTING = {
  title: "Junior Software Engineer - C# Developer",
  description: "We are looking for a Junior Software Engineer to join our growing team. You will work alongside experienced engineers on real projects. No prior commercial experience required - training will be provided. You will learn C#, .NET, SQL Server and Azure. Great opportunity for a recent graduate looking to kick-start their career."
};

function AnalyserTab() {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [result, setResult] = useState(null);

  const loadExample = () => {
    setTitle(EXAMPLE_POSTING.title);
    setDescription(EXAMPLE_POSTING.description);
    setResult(null);
  };

  const clear = () => {
    setTitle(""); setDescription(""); setResult(null);
  };

  const analyse = () => {
    if (!title && !description) return;
    setResult(analyseposting(title, description));
  };

  return (
    <>
      <div style={{
        padding: "20px 24px",
        background: "linear-gradient(135deg, #0c1a2e 0%, #0f172a 100%)",
        border: "1px solid #1e293b", borderRadius: 16, marginBottom: 20,
      }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: "#e2e8f0" }}>
          Live Posting Analyser
        </h1>
        <p style={{ margin: "8px 0 0", fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>
          Paste any UK software engineering job posting (from LinkedIn, Indeed, Reed, or anywhere)
          and get an instant Graduate Accessibility Score (GAS) with category and justification.
          The classifier logic is a direct port of the Python pipeline described in the dissertation.
        </p>
      </div>

      <Card>
        <CardTitle>Enter the posting details</CardTitle>

        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Job title</label>
          <input type="text" value={title} onChange={e => setTitle(e.target.value)}
                 placeholder="e.g. Junior Software Engineer"
                 style={{ ...searchInputStyle, width: "100%" }} />
        </div>

        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Description</label>
          <textarea value={description} onChange={e => setDescription(e.target.value)}
                    placeholder="Paste the full job description here..."
                    style={textAreaStyle} rows={10} />
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={analyse}
                  disabled={!title && !description}
                  style={primaryButton}>
            Analyse posting
          </button>
          <button onClick={loadExample} style={secondaryButton}>Load example</button>
          <button onClick={clear} style={secondaryButton}>Clear</button>
        </div>
      </Card>

      {result && <ResultsPanel result={result} title={title} />}
    </>
  );
}


function ResultsPanel({ result, title }) {
  const { category, gas, components, technologies, years, noExpSignal, degree,
          bootcampMatches, detectedInclusive, detectedExclusive } = result;
  const color = CAT_COLORS[category];

  return (
    <div style={{ marginTop: 20 }}>
      <Section title="Analysis result">
        <Card>
          {/* Hero result */}
          <div style={{
            background: `linear-gradient(135deg, ${color}15 0%, #0f172a 100%)`,
            border: `2px solid ${color}`,
            borderRadius: 12, padding: 24, marginBottom: 16,
            textAlign: "center",
          }}>
            <div style={{ fontSize: 12, color: "#94a3b8", textTransform: "uppercase",
                          letterSpacing: 1, marginBottom: 8 }}>
              Classification
            </div>
            <div style={{ fontSize: 32, fontWeight: 800, color, marginBottom: 6 }}>
              {CAT_LABELS[category]}
            </div>
            <div style={{ fontSize: 14, color: "#cbd5e1", marginBottom: 20 }}>
              {CAT_DESCRIPTIONS[category]}
            </div>
            <div style={{ display: "flex", justifyContent: "center", gap: 40 }}>
              <div>
                <div style={{ fontSize: 44, fontWeight: 800, color, fontFamily: "monospace" }}>
                  {gas}
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8", textTransform: "uppercase" }}>
                  GAS Score
                </div>
              </div>
              <div style={{ borderLeft: "1px solid #1e293b" }} />
              <div style={{ textAlign: "left", fontSize: 12, color: "#cbd5e1", lineHeight: 1.7 }}>
                <div>Technologies: <strong>{technologies.length}</strong></div>
                <div>Years stated: <strong>{years ?? "none detected"}</strong></div>
                <div>No-exp signal: <strong>{noExpSignal ? "yes" : "no"}</strong></div>
                <div>Degree: <strong>{degree ?? "not mentioned"}</strong></div>
              </div>
            </div>
          </div>

          {/* Component breakdown */}
          {components && (
            <div style={{ marginTop: 16 }}>
              <CardTitle>GAS component breakdown</CardTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 8 }}>
                <ComponentBar label="Experience (35%)"       value={components.experience} />
                <ComponentBar label="Degree (20%)"           value={components.degree} />
                <ComponentBar label="Tech breadth (20%)"     value={components.tech_breadth} />
                <ComponentBar label="Language tone (15%)"    value={components.language} />
                <ComponentBar label="Salary transparency (10%)" value={components.salary_transparency} note="default" />
              </div>
            </div>
          )}

          {/* Detected signals */}
          <div style={{ marginTop: 20 }}>
            <CardTitle>Detected signals</CardTitle>
            {technologies.length > 0 && (
              <SignalGroup title="Technologies mentioned" items={technologies} color="#8b5cf6" />
            )}
            {bootcampMatches.length > 0 && (
              <SignalGroup title="Bootcamp signals" items={bootcampMatches} color="#f97316" />
            )}
            {detectedInclusive.length > 0 && (
              <SignalGroup title="Inclusive language (graduate-friendly)" items={detectedInclusive} color="#10b981" />
            )}
            {detectedExclusive.length > 0 && (
              <SignalGroup title="Exclusive language (senior-leaning)" items={detectedExclusive} color="#ef4444" />
            )}
            {technologies.length === 0 && bootcampMatches.length === 0 &&
             detectedInclusive.length === 0 && detectedExclusive.length === 0 && (
              <div style={{ color: "#64748b", fontSize: 12, marginTop: 8 }}>
                No specific signals detected - this typically classifies as "Ambiguous".
              </div>
            )}
          </div>

          {/* Interpretation */}
          <div style={{
            marginTop: 20, padding: "14px 16px",
            background: "#0c1a2e", borderRadius: 8, borderLeft: "3px solid #06b6d4",
            fontSize: 12, color: "#cbd5e1", lineHeight: 1.7,
          }}>
            <strong style={{ color: "#06b6d4" }}>Interpretation:</strong> {interpretResult(result)}
          </div>
        </Card>
      </Section>
    </div>
  );
}

function ComponentBar({ label, value, note }) {
  const color = value >= 75 ? "#10b981" : value >= 50 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ background: "#0f172a", padding: "10px 12px", borderRadius: 8 }}>
      <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 6 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontSize: 22, fontWeight: 800, color, fontFamily: "monospace" }}>{value}</span>
        {note && <span style={{ fontSize: 10, color: "#64748b", fontStyle: "italic" }}>({note})</span>}
      </div>
      <div style={{ height: 3, background: "#1e293b", borderRadius: 2, marginTop: 6, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${value}%`, background: color, transition: "width 0.6s" }} />
      </div>
    </div>
  );
}

function SignalGroup({ title, items, color }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 6 }}>{title}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {items.map((it, i) => (
          <span key={i} style={{
            padding: "3px 8px", borderRadius: 4,
            background: color + "22", color,
            fontSize: 11, fontWeight: 600, fontFamily: "monospace",
          }}>{it}</span>
        ))}
      </div>
    </div>
  );
}

function interpretResult(r) {
  const { category, gas, years, noExpSignal, technologies, isBootcamp, degree } = r;

  if (category === "bootcamp") {
    return "This posting contains multiple signals characteristic of a paid training programme " +
           "rather than a genuine employment offer. Use caution — it may require investment of " +
           "time and money and may not lead directly to a paid role.";
  }
  if (category === "off_topic") {
    return "Despite junior/graduate language in the title, the content suggests this is not a " +
           "software engineering role. Likely a customer service, sales, accounting, or marketing position.";
  }
  if (category === "paradox") {
    return `This posting is labelled 'junior' but demands ${years}+ years of prior experience. ` +
           "This is the graduate paradox - applicants without 2+ years of commercial experience " +
           "will likely be filtered out by automated ATS systems regardless of actual ability.";
  }
  if (category === "genuine_junior") {
    const reason = noExpSignal
      ? "It contains explicit signals welcoming recent graduates (e.g. 'no experience required', 'training provided')"
      : `It states only ${years ?? 0} year(s) of experience required`;
    return `This is a genuine graduate-accessible posting. ${reason}. GAS of ${gas} indicates good accessibility.`;
  }
  return "This posting does not contain explicit experience requirements or graduate-welcoming signals, " +
         "making its accessibility unclear. This is the most common class in the UK market and likely " +
         "reflects the limitations of the 500-character API description truncation.";
}


// ═══════════════════════════════════════════════════════════════════
// SHARED COMPONENTS
// ═══════════════════════════════════════════════════════════════════

function Header({ tab, setTab }) {
  return (
    <div style={headerStyle}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span style={{
          fontSize: 22, fontWeight: 800,
          background: "linear-gradient(90deg,#06b6d4,#8b5cf6)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>GradAccess UK</span>
        <span style={{ fontSize: 12, color: "#64748b" }}>
          Final Year Project · York St John University
        </span>
      </div>
      <div style={{ display: "flex", gap: 4 }}>
        <TabButton active={tab === "overview"} onClick={() => setTab("overview")}>
          Overview
        </TabButton>
        <TabButton active={tab === "analyser"} onClick={() => setTab("analyser")}>
          Live Analyser
        </TabButton>
      </div>
    </div>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      padding: "8px 16px", borderRadius: 8,
      border: active ? "1px solid #06b6d4" : "1px solid #1e293b",
      background: active ? "#06b6d415" : "transparent",
      color: active ? "#06b6d4" : "#94a3b8",
      fontSize: 13, fontWeight: 600, cursor: "pointer",
      transition: "all 0.15s",
    }}>{children}</button>
  );
}

function Intro({ stats }) {
  return (
    <div style={{
      padding: "20px 24px",
      background: "linear-gradient(135deg, #0c1a2e 0%, #0f172a 100%)",
      border: "1px solid #1e293b", borderRadius: 16, marginBottom: 24,
    }}>
      <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: "#e2e8f0" }}>
        Are "Entry-Level" Jobs Really Entry-Level?
      </h1>
      <p style={{ margin: "8px 0 0", fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>
        An empirical analysis of {stats.total_postings} UK software engineering postings
        explicitly labelled "junior", "graduate" or "trainee", collected via the Adzuna public
        API across a 180-day window.
      </p>
    </div>
  );
}

function HeadlineNumbers({ stats }) {
  return (
    <div style={headlineRowStyle}>
      <Headline label="Total postings analysed"       value={stats.total_postings} />
      <Headline label="Bootcamps disguised as jobs"   value={`${stats.bootcamp_disguised_rate_percent}%`} highlight="#f97316" />
      <Headline label="Genuinely graduate-accessible" value={`${stats.genuine_junior_rate_percent}%`} highlight="#10b981" />
      <Headline label="Graduate paradox rate"         value={`${stats.paradox_rate_percent}%`} highlight="#ef4444" />
      <Headline label="Mean GAS (real roles)"         value={stats.gas_mean_real} />
    </div>
  );
}

function Headline({ label, value, highlight }) {
  return (
    <div style={{ ...headlineCardStyle, borderColor: highlight || "#1e293b" }}>
      <div style={{ fontSize: 26, fontWeight: 800, fontFamily: "monospace",
                     color: highlight || "#e2e8f0" }}>{value ?? "—"}</div>
      <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4,
                     textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div style={{ marginTop: 28 }}>
      <h2 style={sectionTitleStyle}>{title}</h2>
      {children}
    </div>
  );
}

function Card({ children }) { return <div style={cardStyle}>{children}</div>; }
function CardTitle({ children }) { return <div style={cardTitleStyle}>{children}</div>; }

function Methodology() {
  return (
    <Section title="Methodology">
      <Card>
        <div style={{ color: "#cbd5e1", fontSize: 13, lineHeight: 1.7 }}>
          <strong style={{ color: "#06b6d4" }}>Data collection.</strong> Adzuna public REST API
          (/v1/api/jobs/gb/search), category=it-jobs, 34 search terms, full-time permanent within 180 days.
          <br/><br/>
          <strong style={{ color: "#06b6d4" }}>NLP pipeline.</strong> Hybrid rule-based + dictionary
          extraction following Khaouja et al. (2021) and Gnehm et al. (2022).
          <br/><br/>
          <strong style={{ color: "#06b6d4" }}>Graduate Accessibility Score.</strong>
          GAS = 0.35·S<sub>exp</sub> + 0.20·S<sub>degree</sub> + 0.20·S<sub>stack</sub>
          + 0.15·S<sub>language</sub> + 0.10·S<sub>salary</sub>. Weights calibrated against
          Fuller &amp; Raman (2021).
          <br/><br/>
          <strong style={{ color: "#06b6d4" }}>Live Analyser.</strong> The classifier on the "Live
          Analyser" tab is a direct JavaScript port of the Python pipeline, identical in regex
          patterns, thresholds and weights.
        </div>
      </Card>
    </Section>
  );
}

function Footer() {
  return (
    <div style={footerStyle}>
      Data source: Adzuna public API (UK) · Methodology documented in Chapters 4-5 of the dissertation ·
      References: Fuller &amp; Raman (2021); Modestino et al. (2020); Khaouja et al. (2021);
      Zhang et al. (2022); Steinmacher et al. (2015).
    </div>
  );
}

function LoadingScreen() {
  return (
    <div style={{ ...pageStyle, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ fontSize: 14, color: "#94a3b8" }}>Loading dataset...</div>
    </div>
  );
}

function ErrorScreen({ message }) {
  return (
    <div style={{ ...pageStyle, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ maxWidth: 500, textAlign: "center" }}>
        <div style={{ fontSize: 16, color: "#f87171", marginBottom: 12 }}>
          Cannot load data files.
        </div>
        <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>
          Make sure <code style={codeStyle}>scored.json</code> and <code style={codeStyle}>
          aggregate_stats.json</code> are in the <code>/public</code> folder.
        </div>
        <div style={{ fontSize: 11, color: "#475569", marginTop: 16, fontFamily: "monospace" }}>
          {message}
        </div>
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════════
// STYLES
// ═══════════════════════════════════════════════════════════════════

const pageStyle = {
  minHeight: "100vh", background: "#020817", color: "#e2e8f0",
  fontFamily: "'Inter', system-ui, sans-serif",
};

const headerStyle = {
  borderBottom: "1px solid #1e293b", padding: "14px 28px",
  display: "flex", justifyContent: "space-between", alignItems: "center",
  position: "sticky", top: 0, background: "#020817", zIndex: 10,
};

const containerStyle = { maxWidth: 1200, margin: "0 auto", padding: "24px 20px" };

const headlineRowStyle = {
  display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))",
  gap: 10, marginBottom: 4,
};

const headlineCardStyle = {
  background: "#0f172a", border: "1px solid #1e293b",
  borderRadius: 12, padding: "16px 18px", textAlign: "center",
};

const sectionTitleStyle = {
  fontSize: 17, fontWeight: 700, color: "#e2e8f0",
  marginBottom: 12, paddingBottom: 8, borderBottom: "1px solid #1e293b",
};

const cardStyle = {
  background: "#0f172a", border: "1px solid #1e293b",
  borderRadius: 12, padding: 20, marginBottom: 14,
};

const cardTitleStyle = {
  fontSize: 12, fontWeight: 700, color: "#94a3b8",
  marginBottom: 14, textTransform: "uppercase", letterSpacing: 0.5,
};

const explainStyle = { fontSize: 11, color: "#64748b", marginTop: 12, lineHeight: 1.6 };
const gridTwo = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 };

const tooltipStyle = {
  background: "#0f172a", border: "1px solid #1e293b",
  borderRadius: 8, fontSize: 12, color: "#e2e8f0",
};

const filtersRow = { display: "flex", gap: 8, marginBottom: 12, alignItems: "center" };

const searchInputStyle = {
  flex: 1, padding: "10px 12px", borderRadius: 8,
  border: "1px solid #1e293b", background: "#020817", color: "#e2e8f0",
  fontSize: 13, outline: "none",
};

const textAreaStyle = {
  width: "100%", padding: "12px 14px", borderRadius: 8,
  border: "1px solid #1e293b", background: "#020817", color: "#e2e8f0",
  fontSize: 13, outline: "none", fontFamily: "inherit", lineHeight: 1.6,
  resize: "vertical", minHeight: 180,
};

const selectStyle = {
  padding: "10px 12px", borderRadius: 8,
  border: "1px solid #1e293b", background: "#020817", color: "#e2e8f0",
  fontSize: 13,
};

const labelStyle = {
  display: "block", fontSize: 11, color: "#94a3b8",
  marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.5,
  fontWeight: 600,
};

const primaryButton = {
  padding: "10px 20px", borderRadius: 8, border: "none",
  background: "linear-gradient(90deg,#06b6d4,#8b5cf6)",
  color: "white", fontWeight: 700, cursor: "pointer", fontSize: 13,
};

const secondaryButton = {
  padding: "10px 16px", borderRadius: 8,
  border: "1px solid #1e293b", background: "transparent",
  color: "#94a3b8", fontWeight: 600, cursor: "pointer", fontSize: 13,
};

const tableWrap = {
  border: "1px solid #1e293b", borderRadius: 8,
  overflow: "auto", maxHeight: 500,
};

const tableStyle = { width: "100%", borderCollapse: "collapse", fontSize: 12 };
const thStyle = {
  textAlign: "left", padding: "10px 12px", background: "#0f172a",
  color: "#94a3b8", fontWeight: 600, position: "sticky", top: 0,
};
const tdStyle = { padding: "10px 12px", color: "#cbd5e1" };
const linkStyle = { color: "#06b6d4", textDecoration: "none" };

const footerStyle = {
  marginTop: 36, padding: "14px 0", borderTop: "1px solid #1e293b",
  fontSize: 10, color: "#475569", textAlign: "center", lineHeight: 1.7,
};

const codeStyle = {
  background: "#1e293b", padding: "2px 6px", borderRadius: 4, fontSize: 11,
};
