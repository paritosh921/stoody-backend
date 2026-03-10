// One-time migration: tenant_db.strokes -> tenant_db.canvas_pages (or another DB if MONGO_DB differs)
//
// Usage:
//   PEN_MONGO_DB_NAME='skb_indl-ciel-1001' MONGO_DB='skb_indl-ciel-1001' mongosh 'YOUR_MONGODB_URI' backend/scripts/migrations/migrate_pen_server_strokes_to_canvas_pages.mongosh.js
//   APPLY_MIGRATION=1 PEN_MONGO_DB_NAME='skb_indl-ciel-1001' MONGO_DB='skb_indl-ciel-1001' mongosh 'YOUR_MONGODB_URI' backend/scripts/migrations/migrate_pen_server_strokes_to_canvas_pages.mongosh.js

const MAIN_DB_NAME = process.env.MONGO_DB || "skillbot_db";
const PEN_DB_NAME = process.env.PEN_MONGO_DB_NAME || MAIN_DB_NAME;
const APPLY_CHANGES = process.env.APPLY_MIGRATION === "1";

function strokeId(stroke) {
  return String(
    stroke?.id ||
    stroke?.strokeId ||
    `legacy:${tojsononeline(stroke)}`
  );
}

function normalizePoints(points) {
  if (!Array.isArray(points)) return [];
  return points.map((pt) => {
    if (pt && typeof pt === "object" && !Array.isArray(pt)) {
      const arr = [pt.x || 0, pt.y || 0, pt.pressure ?? 0.5];
      if ("tiltX" in pt) arr.push(pt.tiltX);
      if ("tiltY" in pt) arr.push(pt.tiltY);
      if ("timestamp" in pt) arr.push(pt.timestamp);
      return arr;
    }
    return pt;
  });
}

function normalizeStroke(raw, bookType, pageNumber) {
  return {
    id: String(raw?.id || raw?.strokeId || `legacy:${tojsononeline(raw)}`),
    points: normalizePoints(raw?.points || []),
    strokeWidth: Number(raw?.strokeWidth ?? 1.3),
    color: raw?.color || "#000000",
    tool: raw?.tool || "pen",
    timestamp: raw?.timestamp,
    svgPath: raw?.svgPath,
    baseWidthMm: raw?.baseWidthMm,
    sourceMode: raw?.sourceMode,
    startedAt: raw?.startedAt,
    endedAt: raw?.endedAt,
    pageNumber: raw?.pageNumber ?? pageNumber,
    bookType: raw?.bookType ?? bookType,
    penMac: raw?.penMac || raw?.pen_mac || null,
  };
}

function mergeUnique(existing, incoming) {
  const ordered = Array.isArray(existing) ? existing.slice() : [];
  const seen = new Set(ordered.map((s) => strokeId(s)));
  for (const stroke of incoming || []) {
    const sid = strokeId(stroke);
    if (seen.has(sid)) continue;
    seen.add(sid);
    ordered.push(stroke);
  }
  return ordered;
}

const mainDb = db.getSiblingDB(MAIN_DB_NAME);
const penDb = db.getSiblingDB(PEN_DB_NAME);
const penStrokes = penDb.strokes;
const canvasPages = mainDb.canvas_pages;

print(`Main DB: ${MAIN_DB_NAME}`);
print(`Pen DB: ${PEN_DB_NAME}`);
print(`Mode: ${APPLY_CHANGES ? "APPLY" : "DRY RUN"}`);
print("");

const groups = penStrokes.aggregate([
  {
    $match: {
      user_id: { $exists: true, $ne: null },
      book_type: { $exists: true, $ne: null },
      page_number: { $exists: true, $ne: null },
    },
  },
  { $sort: { timestamp: 1 } },
  {
    $group: {
      _id: {
        user_id: "$user_id",
        book_type: { $toUpper: "$book_type" },
        page_number: "$page_number",
      },
      docs: { $push: "$$ROOT" },
      first_ts: { $min: "$timestamp" },
      last_ts: { $max: "$timestamp" },
    },
  },
]).toArray();

print(`Discovered ${groups.length} user/book/page groups from pen-server strokes.`);

let migrated = 0;
let skipped = 0;

for (const group of groups) {
  const userId = String(group._id.user_id);
  const bookType = String(group._id.book_type || "").toUpperCase();
  const pageNumber = Number(group._id.page_number);
  const docs = group.docs || [];

  let mergedLegacyStrokes = [];
  let sessionId = null;
  let penMac = null;
  let pageStyle = null;
  let canvasBackground = null;

  for (const doc of docs) {
    if (!sessionId && doc.session_id) sessionId = doc.session_id;
    if (!penMac && doc.pen_mac) penMac = doc.pen_mac;
    if (!pageStyle && doc.page_style) pageStyle = doc.page_style;
    if (!canvasBackground && doc.canvas_background) canvasBackground = doc.canvas_background;

    for (const stroke of doc.strokes || []) {
      mergedLegacyStrokes.push(normalizeStroke(stroke, bookType, pageNumber));
    }
  }

  if (!mergedLegacyStrokes.length) {
    skipped += 1;
    continue;
  }

  const key = {
    user_id: userId,
    book_type: bookType,
    page_number: pageNumber,
  };

  const existing = canvasPages.findOne(key);
  const existingStrokes = existing?.strokes || [];
  const finalStrokes = mergeUnique(existingStrokes, mergedLegacyStrokes);

  if (finalStrokes.length === existingStrokes.length) {
    skipped += 1;
    continue;
  }

  const now = new Date();
  const doc = {
    user_id: userId,
    admin_id: existing?.admin_id || null,
    book_type: bookType,
    page_number: pageNumber,
    strokes: finalStrokes,
    page_style: existing?.page_style || pageStyle || null,
    canvas_background: existing?.canvas_background || canvasBackground || null,
    stroke_count: finalStrokes.length,
    pen_mac: existing?.pen_mac || penMac || null,
    source: existing?.source || "pen_server_migration",
    last_modified: now,
    client_last_modified: existing?.client_last_modified || (group.last_ts ? group.last_ts.getTime() : null),
    version: Number(existing?.version || 0) + 1,
    session_id: existing?.session_id || sessionId || null,
    first_activity: existing?.first_activity || (group.first_ts ? group.first_ts.getTime() : null),
    last_activity: Math.max(Number(existing?.last_activity || 0), group.last_ts ? group.last_ts.getTime() : 0) || null,
  };

  if (APPLY_CHANGES) {
    canvasPages.replaceOne(key, doc, { upsert: true });
  }

  migrated += 1;
  print(`${APPLY_CHANGES ? "Migrated" : "Would migrate"} user=${userId} book=${bookType} page=${pageNumber} legacy=${mergedLegacyStrokes.length} existing=${existingStrokes.length} final=${finalStrokes.length}`);
}

print("");
print(`${APPLY_CHANGES ? "Migration" : "Dry-run"} complete.`);
print(`Migrated groups: ${migrated}`);
print(`Skipped groups:  ${skipped}`);
