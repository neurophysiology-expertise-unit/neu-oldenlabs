-- Force pandoc's LaTeX writer to render every table with explicit,
-- content-proportional column widths, instead of the pipe-table default.
--
-- A markdown pipe table always parses with width 0 ("auto") per column,
-- which the LaTeX writer renders as a plain `l` column: sized to its
-- widest cell with NO WRAPPING. A table with several columns and any
-- reasonably long header or cell (this project's `cages_contributing`
-- header, or a measure name like `occupancy_front_half_pct_light`) then
-- overflows the page margin outright -- verified in report.pdf before this
-- filter existed: the `ci_high` column ran off the right edge entirely.
-- Pandoc has its own heuristic that SOMETIMES switches a wide pipe table to
-- wrapped columns, but it triggers inconsistently (depends on a rough
-- character-count guess against its `--columns` prose-wrap width, nothing
-- to do with the actual LaTeX page width) and, when it does trigger,
-- splits every column EQUALLY regardless of content -- cramming a long
-- `measure` column into the same width as a one-character `n_wt` column.
--
-- This filter instead always assigns width proportional to each column's
-- own widest cell (header included), scaled so the row totals a fixed,
-- safely-under-100% fraction of the text width -- guaranteeing pandoc
-- emits wrapped (p{width}) columns, in proportions that actually reflect
-- the content, for every table, not just the ones its own heuristic
-- happens to flag.
local function cell_text(cell)
    local parts = {}
    for _, block in ipairs(cell) do
        if block.t == "Plain" or block.t == "Para" then
            for _, inline in ipairs(block.content) do
                if inline.t == "Str" then
                    table.insert(parts, inline.text)
                elseif inline.t == "Code" then
                    table.insert(parts, inline.text)
                end
            end
        end
    end
    return table.concat(parts, " ")
end

-- A flat character COUNT is not a proportional-width estimate: digits and
-- capitals render noticeably wider in Latin Modern Roman than lowercase
-- letters, punctuation, or underscores do. `difference` (10 lowercase
-- letters) and `-0.007103` (9 digits/symbols) measured EQUAL under a flat
-- count, but the digit string rendered wider -- confirmed by an Overfull
-- \hbox on exactly that column once real report content (not a
-- one-row synthetic test) was rendered. These weights approximate each
-- character class's advance width in em, relative to a lowercase letter.
local WEIGHTS = {digit = 1.35, upper = 1.05, lower = 0.85, punct = 0.85,
                 underscore = 0.6, space = 0.5}

local function char_weight(ch)
    if ch:match("%d") then
        return WEIGHTS.digit
    elseif ch:match("%u") then
        return WEIGHTS.upper
    elseif ch:match("%l") then
        return WEIGHTS.lower
    elseif ch == "_" then
        return WEIGHTS.underscore
    elseif ch == " " then
        return WEIGHTS.space
    else
        return WEIGHTS.punct
    end
end

local function cell_width(cell)
    local text = cell_text(cell)
    local width = 0
    for _, codepoint in utf8.codes(text) do
        width = width + char_weight(utf8.char(codepoint))
    end
    return width
end

-- Fixed per-column padding (in weighted "characters"), added BEFORE
-- normalising. pandoc/xelatex renders each cell inside a minipage sized in
-- exact proportion to its estimated width, and a minipage even a few
-- points narrower than its own content produces an Overfull \hbox: LaTeX
-- does not reject or visibly warn in the rendered page, it just lets the
-- text spill past the box's right edge into the next column, which reads
-- as two columns' text colliding. This padding is a further safety margin
-- on top of the weighting above; it matters far more, proportionally, for
-- a short column (`n_wt`) than a long one (`measure`), which is why it is
-- added per-column before scaling rather than as one global fudge factor.
local COLUMN_PADDING = 2

-- None of this project's table cells contain a space -- every value is a
-- single identifier (`occupancy_front_half_pct_light`) or a bare number.
-- LaTeX's `\raggedright` wraps at spaces, not at `_`, so an identifier
-- longer than its assigned column width has NOWHERE to break: it overflows
-- the minipage box outright and bleeds into the next column, no matter how
-- generous the width estimate above is (confirmed: even the widest
-- generous-side estimate still overflowed on `social_distance_cm_all`).
-- This inserts an explicit (invisible, no hyphen drawn) break opportunity
-- after every underscore so identifiers can actually wrap.
local function breakable_inlines(text)
    local result = {}
    local start = 1
    while true do
        local pos = text:find("_", start, true)
        if not pos then
            local rest = text:sub(start)
            if #rest > 0 then
                table.insert(result, pandoc.Str(rest))
            end
            break
        end
        table.insert(result, pandoc.Str(text:sub(start, pos)))
        table.insert(result, pandoc.RawInline("latex", "\\allowbreak "))
        start = pos + 1
    end
    return result
end

local function add_break_points(cell)
    for _, block in ipairs(cell) do
        if block.t == "Plain" or block.t == "Para" then
            local new_content = {}
            for _, inline in ipairs(block.content) do
                if inline.t == "Str" and inline.text:find("_", 1, true) then
                    for _, piece in ipairs(breakable_inlines(inline.text)) do
                        table.insert(new_content, piece)
                    end
                else
                    table.insert(new_content, inline)
                end
            end
            block.content = new_content
        end
    end
    return cell
end

function Table(tbl)
    local ncol = #tbl.headers
    if ncol == 0 then
        return tbl
    end

    local widths = {}
    for i = 1, ncol do
        widths[i] = math.max(cell_width(tbl.headers[i]), 1)
    end
    for _, row in ipairs(tbl.rows) do
        for i = 1, ncol do
            local width = cell_width(row[i])
            if width > widths[i] then
                widths[i] = width
            end
        end
    end
    for i = 1, ncol do
        widths[i] = widths[i] + COLUMN_PADDING
    end

    local total = 0
    for i = 1, ncol do
        total = total + widths[i]
    end
    if total == 0 then
        return tbl
    end

    -- 0.94, not 1.0: longtable needs headroom for inter-column padding and
    -- rule lines that the weighted estimate above does not account for: a
    -- row that totals exactly 100% of the text width still overflows once
    -- that padding is added.
    local scale = 0.94 / total
    for i = 1, ncol do
        tbl.widths[i] = widths[i] * scale
        tbl.headers[i] = add_break_points(tbl.headers[i])
    end
    for _, row in ipairs(tbl.rows) do
        for i = 1, ncol do
            row[i] = add_break_points(row[i])
        end
    end
    return tbl
end
