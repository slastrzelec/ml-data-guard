// Front end of ml-data-guard. Server answers are written with textContent only, never as markup.
(function () {
  "use strict";

  var RULES = [
    ["Distances, limit calibrated on clean batches", [
      ["calibrated:linf", "L-inf"], ["calibrated:l1", "L1"], ["calibrated:kl", "KL"],
      ["calibrated:js", "JS"], ["calibrated:psi", "PSI"], ["calibrated:chi2", "chi-square"],
      ["calibrated:ks", "KS"]]],
    ["Fixed thresholds from tutorials", [
      ["textbook:chi2 p<0.05", "chi-square p < 0.05"], ["textbook:ks p<0.05", "KS p < 0.05"],
      ["textbook:linf>0.01", "L-inf > 0.01"], ["textbook:psi>0.1", "PSI > 0.1"],
      ["textbook:psi>0.2", "PSI > 0.2"]]],
    ["Schema check", [["schema", "Inferred schema"]]]
  ];
  var LABELS = {
    pinned_value: "Value pinned to -1", dropped_feature: "Dropped column",
    type_change: "Number stored as text", new_category: "New category",
    out_of_domain_numeric: "Out-of-range values", missing_rate_increase: "Missing values",
    unit_change: "Unit change (x1000)", mean_shift: "Mean shift", variance_change: "Variance change",
    duplicate_rows: "Duplicated rows", tiny_noise: "Tiny noise (negative control)"
  };
  var DEFAULTS = {
    fraction_of_rows: 0.01, sigma: 0.25, relative_scale: 0.25, binary: 1
  };
  var UNITS = {
    fraction_of_rows: "(share of rows, 0.0001 to 1)", sigma: "(standard deviations, 0 to 5)",
    relative_scale: "(growth of the spread, 0 to 5)", binary: "(the whole column)"
  };

  function byId(id) { return document.getElementById(id); }
  function node(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined && text !== null) { e.textContent = String(text); }
    return e;
  }
  function clear(e) { while (e.firstChild) { e.removeChild(e.firstChild); } }
  function option(select, value, text, selected) {
    var o = node("option", null, text);
    o.value = String(value);
    o.selected = !!selected;
    select.appendChild(o);
  }
  function num(x) {
    if (x === null || x === undefined) { return "-"; }
    var a = Math.abs(x);
    return a !== 0 && (a < 0.001 || a >= 1e5) ? x.toExponential(2) : String(Number(x.toPrecision(4)));
  }

  async function call(url, options) {
    var response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      throw new Error("the service did not answer; if it was asleep, try again in a minute");
    }
    var body = null;
    try { body = await response.json(); } catch (error) { body = null; }
    if (!response.ok) {
      throw new Error(body && body.error ? body.error : "request failed (" + response.status + ")");
    }
    return body;
  }

  // ------------------------------------------------------------------ tabs
  function select(tab) {
    ["demo", "file"].forEach(function (name) {
      var on = name === tab;
      byId("tab-" + name).setAttribute("aria-selected", on ? "true" : "false");
      byId("panel-" + name).hidden = !on;
    });
  }
  byId("tab-demo").addEventListener("click", function () { select("demo"); });
  byId("tab-file").addEventListener("click", function () { select("file"); });

  // ------------------------------------------------------------------ demo
  var options = null;

  function currentFault() {
    var name = byId("demo-fault").value;
    return options.faults.filter(function (f) { return f.name === name; })[0];
  }

  function faultChanged() {
    var fault = currentFault(), feature = byId("demo-feature"), severity = byId("demo-severity");
    clear(feature);
    if (fault.features.length) {
      option(feature, "", "random", true);
      fault.features.forEach(function (name) { option(feature, name, name); });
      feature.disabled = false;
    } else {
      option(feature, "", "whole rows", true);
      feature.disabled = true;
    }
    severity.value = String(DEFAULTS[fault.unit]);
    severity.min = String(fault.range[0]);
    severity.max = String(fault.range[1]);
    severity.disabled = fault.unit === "binary";
    byId("demo-unit").textContent = UNITS[fault.unit] || "";
  }

  function badge(verdict) {
    if (!verdict || verdict.applies === false) { return node("span", "badge na", "n/a"); }
    if (verdict.alarm === null) { return node("span", "badge na", "column gone"); }
    return verdict.alarm ? node("span", "badge alarm", "alarm") : node("span", "badge quiet", "quiet");
  }

  function scoreText(verdict) {
    if (!verdict || verdict.applies === false || verdict.score === undefined) { return ""; }
    return num(verdict.score) + (verdict.limit !== undefined ? " / " + num(verdict.limit) : "");
  }

  function otherAlarms(result, side, rule) {
    var count = 0;
    result.features.forEach(function (name) {
      if (name === result.feature) { return; }
      var v = result[side][name][rule];
      if (v && v.alarm === true) { count += 1; }
    });
    return count;
  }

  function renderDemo(result) {
    var box = byId("demo-result");
    clear(box);
    var summary = node("p", "summary");
    var fault = LABELS[result.fault] || result.fault;
    var where = result.feature ? " on feature " + result.feature : " on whole rows";
    summary.appendChild(node("strong", null, fault + where + ". "));
    summary.appendChild(document.createTextNode(
      result.batch_size.toLocaleString("en-US") + " rows per batch, seed " + result.seed + ". " +
      (result.negative_control ? "This is a negative control: every alarm is a false alarm. " : "") +
      "Score / limit is shown for the faulted feature; the last column counts alarms on the other " +
      (result.features.length - (result.feature ? 1 : 0)) + " features of the faulty batch."));
    box.appendChild(summary);

    var table = node("table"), head = node("tr");
    ["Rule", "Clean batch", "Faulty batch", "Score / limit (faulty)", "Other features"].forEach(function (t, i) {
      head.appendChild(node("th", i > 2 ? "num" : null, t));
    });
    var thead = node("thead"); thead.appendChild(head); table.appendChild(thead);
    var body = node("tbody");
    RULES.forEach(function (group) {
      var g = node("tr", "group"), cell = node("td", null, group[0]);
      cell.colSpan = 5; g.appendChild(cell); body.appendChild(g);
      group[1].forEach(function (rule) {
        var tr = node("tr");
        tr.appendChild(node("td", null, rule[1]));
        var clean = node("td"), faulty = node("td");
        if (result.feature) {
          clean.appendChild(badge(result.clean[result.feature][rule[0]]));
          faulty.appendChild(badge(result.faulty[result.feature][rule[0]]));
        } else {
          var c = otherAlarms(result, "clean", rule[0]), f = otherAlarms(result, "faulty", rule[0]);
          clean.appendChild(node("span", c ? "badge alarm" : "badge quiet", c + " feature(s)"));
          faulty.appendChild(node("span", f ? "badge alarm" : "badge quiet", f + " feature(s)"));
        }
        tr.appendChild(clean);
        tr.appendChild(faulty);
        tr.appendChild(node("td", "num", result.feature ? scoreText(result.faulty[result.feature][rule[0]]) : ""));
        tr.appendChild(node("td", "num", result.feature ? String(otherAlarms(result, "faulty", rule[0])) : ""));
        body.appendChild(tr);
      });
    });
    table.appendChild(body);
    var scroll = node("div", "scroll"); scroll.appendChild(table); box.appendChild(scroll);
    var types = result.schema_anomaly_types;
    box.appendChild(node("p", "muted",
      "Schema findings - clean batch: " + (types.clean.join(", ") || "none") +
      "; faulty batch: " + (types.faulty.join(", ") || "none") + "."));
  }

  async function runDemo() {
    var status = byId("demo-status"), button = byId("demo-run");
    button.disabled = true;
    status.textContent = "Calibrating and checking...";
    status.className = "muted";
    try {
      var feature = byId("demo-feature").value;
      var result = await call("/api/v1/demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fault: byId("demo-fault").value,
          severity: Number(byId("demo-severity").value),
          batch_size: Number(byId("demo-size").value),
          seed: Number(byId("demo-seed").value || 0),
          feature: feature || null
        })
      });
      renderDemo(result);
      status.textContent = "";
    } catch (error) {
      status.textContent = error.message;
      status.className = "error";
    } finally {
      button.disabled = false;
    }
  }

  async function initDemo() {
    try {
      options = await call("/api/v1/demo/options");
    } catch (error) {
      byId("demo-status").textContent = error.message;
      byId("demo-status").className = "error";
      return;
    }
    options.faults.forEach(function (f) {
      option(byId("demo-fault"), f.name, LABELS[f.name] || f.name, f.name === "unit_change");
    });
    options.batch_sizes.forEach(function (s) {
      option(byId("demo-size"), s, s.toLocaleString("en-US"), s === 1000);
    });
    faultChanged();
    byId("demo-fault").addEventListener("change", faultChanged);
    byId("demo-run").addEventListener("click", runDemo);
  }

  // ------------------------------------------------------------------ own file
  var schema = null;

  function setSchema(value, text) {
    schema = value;
    byId("file-download").disabled = !schema;
    byId("file-validate").disabled = !schema;
    byId("file-schema-status").textContent = text;
    byId("file-schema-status").className = "muted";
  }

  function fail(id, error) {
    byId(id).textContent = error.message;
    byId(id).className = "error";
  }

  async function inferSchema() {
    var file = byId("file-reference").files[0];
    if (!file) { fail("file-schema-status", new Error("choose a reference CSV first")); return; }
    var form = new FormData();
    form.append("data", file);
    form.append("exclude", byId("file-exclude").value);
    byId("file-schema-status").textContent = "Inferring...";
    try {
      var result = await call("/api/v1/infer", { method: "POST", body: form });
      setSchema(result.schema, "Schema of " + result.schema.features.length + " feature(s) from " +
        result.n_rows.toLocaleString("en-US") + " rows.");
    } catch (error) {
      fail("file-schema-status", error);
    }
  }

  function loadSchemaFile() {
    var file = byId("file-schema").files[0];
    if (!file) { return; }
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var parsed = JSON.parse(String(reader.result));
        if (!parsed || !Array.isArray(parsed.features)) { throw new Error("not a schema file"); }
        setSchema(parsed, "Loaded a schema of " + parsed.features.length + " feature(s).");
      } catch (error) {
        fail("file-schema-status", new Error("the file is not an ml-data-guard schema"));
      }
    };
    reader.readAsText(file);
  }

  function downloadSchema() {
    if (!schema) { return; }
    var blob = new Blob([JSON.stringify(schema, null, 2) + "\n"], { type: "application/json" });
    var link = node("a");
    link.href = URL.createObjectURL(blob);
    link.download = "schema.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  }

  function renderReport(title, report) {
    var box = node("div");
    var ok = report.status === "ok";
    var head = node("h2", null, title + ": ");
    head.appendChild(node("span", ok ? "badge quiet" : "badge alarm", ok ? "passed" : "anomalies"));
    box.appendChild(head);
    if (!report.anomalies.length) {
      box.appendChild(node("p", "muted", "No anomaly in " + report.n_rows.toLocaleString("en-US") + " rows."));
      return box;
    }
    var list = node("ul", "anomalies");
    report.anomalies.forEach(function (a) {
      var li = node("li");
      li.appendChild(node("span", a.level === "error" ? "badge alarm" : "badge warning", a.level));
      li.appendChild(document.createTextNode(" " + (a.feature || "(batch)") + ": " + a.type + " - " + a.description));
      if (a.suggestion) { li.appendChild(node("div", "suggestion", "Suggestion: " + a.suggestion)); }
      list.appendChild(li);
    });
    box.appendChild(list);
    return box;
  }

  async function checkBatch() {
    var file = byId("file-batch").files[0], box = byId("file-result");
    if (!file) { fail("file-status", new Error("choose a batch CSV first")); return; }
    var compare = byId("file-compare").checked, reference = byId("file-reference").files[0];
    if (compare && !reference) { fail("file-status", new Error("comparing needs the reference CSV of step 1")); return; }
    byId("file-validate").disabled = true;
    byId("file-status").textContent = compare ? "Checking and calibrating..." : "Checking...";
    byId("file-status").className = "muted";
    clear(box);
    try {
      var form = new FormData();
      form.append("data", file);
      form.append("schema", JSON.stringify(schema));
      box.appendChild(renderReport("Schema check", await call("/api/v1/validate", { method: "POST", body: form })));
      if (compare) {
        var both = new FormData();
        both.append("reference", reference);
        both.append("data", file);
        both.append("schema", JSON.stringify(schema));
        box.appendChild(renderReport("Distribution shift (L-inf, calibrated)", await call("/api/v1/compare", { method: "POST", body: both })));
      }
      byId("file-status").textContent = "";
    } catch (error) {
      fail("file-status", error);
    } finally {
      byId("file-validate").disabled = !schema;
    }
  }

  byId("file-infer").addEventListener("click", inferSchema);
  byId("file-schema").addEventListener("change", loadSchemaFile);
  byId("file-download").addEventListener("click", downloadSchema);
  byId("file-validate").addEventListener("click", checkBatch);
  initDemo();
})();
