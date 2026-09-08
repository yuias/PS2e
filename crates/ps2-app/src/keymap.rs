//! The keyboard binding dialog: a DualShock 2 diagram with one label box
//! per digital button, armed by a click and bound by the next keypress.
//!
//! The controller is painted rather than loaded from an image. It costs no
//! decoder and no megabyte in the binary, it scales to whatever size the
//! window is dragged to, it takes its colours from the egui theme so it is
//! readable in light and dark alike, and the button being bound can be lit
//! up — none of which a bitmap gives for free.

use crate::config::KeyBindings;
use eframe::egui;

/// Button names in [`KeyBindings::pairs`] order, which is the order every
/// table in this module indexes by.
pub const BUTTON_NAMES: [&str; 16] = [
    "up", "down", "left", "right", "cross", "circle", "square", "triangle", "L1", "L2", "R1", "R2",
    "L3", "R3", "start", "select",
];

/// The whole diagram is laid out in this fixed space and scaled to fit the
/// dialog, so the layout below can be written in round numbers.
const CANVAS: egui::Vec2 = egui::vec2(1240.0, 690.0);

/// Where the controller drawing sits inside [`CANVAS`]. The drawing has its
/// own 900x520 coordinate space; the margin left over is what the label
/// boxes and their leader lines live in.
const ORIGIN: egui::Vec2 = egui::vec2(170.0, 60.0);

/// Label box size, in canvas units.
const BOX: egui::Vec2 = egui::vec2(140.0, 30.0);

/// Line weight of the drawing, in canvas units.
const LINE: f32 = 3.0;

/// Maps canvas coordinates onto the screen rect the dialog handed us.
#[derive(Clone, Copy)]
struct View {
    origin: egui::Pos2,
    scale: f32,
}

impl View {
    /// Fit [`CANVAS`] into `rect`, centred, without distorting it.
    fn fit(rect: egui::Rect) -> Self {
        let scale = (rect.width() / CANVAS.x).min(rect.height() / CANVAS.y);
        let used = CANVAS * scale;
        Self { origin: rect.center() - used / 2.0, scale }
    }

    /// A point in canvas coordinates.
    fn at(self, x: f32, y: f32) -> egui::Pos2 {
        self.origin + egui::vec2(x, y) * self.scale
    }

    /// A point in the controller drawing's own coordinates.
    fn art(self, x: f32, y: f32) -> egui::Pos2 {
        self.at(x + ORIGIN.x, y + ORIGIN.y)
    }

    fn len(self, v: f32) -> f32 {
        v * self.scale
    }

    fn stroke(self, width: f32, color: egui::Color32) -> egui::Stroke {
        egui::Stroke::new(self.len(width).max(1.0), color)
    }
}

/// A cubic path in the drawing's coordinates, flattened as it is built.
struct Path {
    view: View,
    points: Vec<egui::Pos2>,
    cur: (f32, f32),
}

impl Path {
    fn new(view: View, x: f32, y: f32) -> Self {
        Self { view, points: vec![view.art(x, y)], cur: (x, y) }
    }

    fn line(&mut self, x: f32, y: f32) {
        self.cur = (x, y);
        self.points.push(self.view.art(x, y));
    }

    /// Twelve segments per curve: smooth at every size this dialog is drawn
    /// at, and it keeps the whole body outline under 200 points.
    fn cubic(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        const STEPS: usize = 12;
        let (x0, y0) = self.cur;
        for i in 1..=STEPS {
            let t = i as f32 / STEPS as f32;
            let u = 1.0 - t;
            let bx = u * u * u * x0 + 3.0 * u * u * t * x1 + 3.0 * u * t * t * x2 + t * t * t * x;
            let by = u * u * u * y0 + 3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t * t * t * y;
            self.points.push(self.view.art(bx, by));
        }
        self.cur = (x, y);
    }
}

/// The outline of a pressable part, in the drawing's coordinates.
#[derive(Clone, Copy)]
enum Art {
    /// `x`/`y` is the top-left corner, as in the source drawing.
    Rect { x: f32, y: f32, w: f32, h: f32, r: f32 },
    Circle { x: f32, y: f32, r: f32 },
    Tri([(f32, f32); 3]),
    /// The shoulder buttons: a rectangle with the two top corners rounded
    /// off, drawn as an explicit path because the shape is asymmetric.
    Shoulder { x: f32, y: f32, w: f32, h: f32 },
}

impl Art {
    fn paint(self, p: &egui::Painter, view: View, fill: egui::Color32, stroke: egui::Stroke) {
        match self {
            Art::Rect { x, y, w, h, r } => {
                let rect = egui::Rect::from_min_max(view.art(x, y), view.art(x + w, y + h));
                let cr = egui::CornerRadius::same(view.len(r).round().clamp(0.0, 255.0) as u8);
                p.rect_filled(rect, cr, fill);
                p.rect_stroke(rect, cr, stroke, egui::StrokeKind::Middle);
            }
            Art::Circle { x, y, r } => {
                p.circle(view.art(x, y), view.len(r), fill, stroke);
            }
            Art::Tri(pts) => {
                let pts: Vec<_> = pts.iter().map(|&(x, y)| view.art(x, y)).collect();
                p.add(egui::Shape::convex_polygon(pts, fill, stroke));
            }
            Art::Shoulder { x, y, w, h } => {
                // The corners round off over the top half, then the sides
                // splay out to the bottom edge.
                let (mid, bottom) = (y + h * 0.53, y + h);
                let mut path = Path::new(view, x + 44.0, y);
                path.line(x + w - 44.0, y);
                path.cubic(x + w - 24.0, y, x + w - 8.0, y + h * 0.24, x + w - 6.0, mid);
                path.line(x + w, bottom);
                path.line(x, bottom);
                path.line(x + 6.0, mid);
                path.cubic(x + 8.0, y + h * 0.24, x + 24.0, y, x + 44.0, y);
                p.add(egui::Shape::convex_polygon(path.points, fill, stroke));
            }
        }
    }
}

/// The mark printed on a button, once the button itself is drawn.
#[derive(Clone, Copy)]
enum Mark {
    None,
    /// Text inside the button (the shoulder buttons).
    Inside(&'static str),
    Triangle,
    Circle,
    Cross,
    Square,
}

/// How a leader line leaves its label box: horizontal first, or vertical
/// first. Either way it turns exactly once before reaching the button.
#[derive(Clone, Copy)]
enum Route {
    Horizontal,
    Vertical,
}

/// One bindable button on the diagram.
struct Slot {
    /// Index into [`BUTTON_NAMES`] and [`KeyBindings::pairs`].
    idx: usize,
    art: Art,
    mark: Mark,
    /// Where the leader line meets the button, in drawing coordinates.
    anchor: (f32, f32),
    /// Centre of the label box, in canvas coordinates.
    label: (f32, f32),
    route: Route,
    /// Canvas coordinate the leader turns at. `None` turns level with the
    /// anchor, which is what all but two of them can do; the rest would take
    /// the corner through another button.
    elbow: Option<f32>,
    /// Drawn before the two rings, the way the shoulders sit behind them on
    /// the controller. Everything else goes on top.
    behind: bool,
}

const fn slot(
    idx: usize,
    art: Art,
    mark: Mark,
    anchor: (f32, f32),
    label: (f32, f32),
    route: Route,
) -> Slot {
    Slot { idx, art, mark, anchor, label, route, elbow: None, behind: false }
}

const fn elbow(s: Slot, at: f32) -> Slot {
    Slot { elbow: Some(at), ..s }
}

const fn behind(s: Slot) -> Slot {
    Slot { behind: true, ..s }
}

/// The layout. Anchors sit on the edge of the button the line comes in
/// from, so every leader turns at most once and none of them ends inside
/// the shape it points at.
const SLOTS: [Slot; 16] = [
    // D-pad, labelled down the left margin.
    slot(0, Art::Rect { x: 180.0, y: 160.0, w: 50.0, h: 62.0, r: 10.0 }, Mark::None, (180.0, 191.0), (85.0, 240.0), Route::Horizontal),
    slot(1, Art::Rect { x: 180.0, y: 238.0, w: 50.0, h: 62.0, r: 10.0 }, Mark::None, (180.0, 269.0), (85.0, 345.0), Route::Horizontal),
    slot(2, Art::Rect { x: 132.0, y: 208.0, w: 62.0, h: 50.0, r: 10.0 }, Mark::None, (132.0, 233.0), (85.0, 293.0), Route::Horizontal),
    // Turning level with the anchor would put the corner in the left stick.
    elbow(slot(3, Art::Rect { x: 216.0, y: 208.0, w: 62.0, h: 50.0, r: 10.0 }, Mark::None, (247.0, 258.0), (85.0, 400.0), Route::Horizontal), 410.0),
    // Face buttons, labelled down the right margin. Every leader comes in on
    // the side facing the margin, so none of them ends inside its button.
    slot(4, Art::Circle { x: 695.0, y: 295.0, r: 33.0 }, Mark::Cross, (728.0, 295.0), (1155.0, 355.0), Route::Horizontal),
    slot(5, Art::Circle { x: 760.0, y: 230.0, r: 33.0 }, Mark::Circle, (793.0, 230.0), (1155.0, 290.0), Route::Horizontal),
    // Square is the far side of the cluster: its leader turns in the gap
    // between the right stick and cross.
    elbow(slot(6, Art::Circle { x: 630.0, y: 230.0, r: 33.0 }, Mark::Square, (663.0, 230.0), (1155.0, 400.0), Route::Horizontal), 827.0),
    slot(7, Art::Circle { x: 695.0, y: 165.0, r: 33.0 }, Mark::Triangle, (728.0, 165.0), (1155.0, 232.0), Route::Horizontal),
    // Shoulders: L1/R1 continue the side columns, L2/R2 go along the top.
    // They stop at y=130, which keeps L1 off the d-pad and R1 off triangle
    // while still meeting the body's top edge at y=120.
    behind(slot(8, Art::Shoulder { x: 106.0, y: 68.0, w: 192.0, h: 62.0 }, Mark::Inside("L1"), (106.0, 130.0), (85.0, 190.0), Route::Horizontal)),
    behind(slot(9, Art::Rect { x: 142.0, y: 20.0, w: 120.0, h: 50.0, r: 24.0 }, Mark::Inside("L2"), (202.0, 20.0), (372.0, 30.0), Route::Vertical)),
    behind(slot(10, Art::Shoulder { x: 602.0, y: 68.0, w: 192.0, h: 62.0 }, Mark::Inside("R1"), (794.0, 130.0), (1155.0, 190.0), Route::Horizontal)),
    behind(slot(11, Art::Rect { x: 638.0, y: 20.0, w: 120.0, h: 50.0, r: 24.0 }, Mark::Inside("R2"), (698.0, 20.0), (868.0, 30.0), Route::Vertical)),
    // Stick clicks, below the sticks they belong to.
    slot(12, Art::Circle { x: 315.0, y: 350.0, r: 66.0 }, Mark::None, (315.0, 416.0), (380.0, 640.0), Route::Vertical),
    slot(13, Art::Circle { x: 585.0, y: 350.0, r: 66.0 }, Mark::None, (585.0, 416.0), (860.0, 640.0), Route::Vertical),
    // Start and select reach up from the top margin, between the shoulders.
    slot(14, Art::Tri([(500.0, 192.0), (550.0, 207.0), (500.0, 222.0)]), Mark::None, (525.0, 192.0), (720.0, 30.0), Route::Vertical),
    slot(15, Art::Rect { x: 335.0, y: 188.0, w: 54.0, h: 30.0, r: 6.0 }, Mark::None, (362.0, 188.0), (545.0, 30.0), Route::Vertical),
];

/// The dialog's own state: the bindings being edited, and which button is
/// waiting for a keypress. The caller keeps the live bindings untouched
/// until [`Outcome::Accept`].
pub struct Binder {
    draft: KeyBindings,
    /// Index into [`SLOTS`]' `idx` space, set by clicking a label box.
    armed: Option<usize>,
}

/// What the dialog wants done once a frame of it has been drawn.
pub enum Outcome {
    /// Still open; nothing to do.
    Open,
    Accept(KeyBindings),
    Cancel,
}

impl Binder {
    pub fn new(keys: &KeyBindings) -> Self {
        Self { draft: keys.clone(), armed: None }
    }

    /// True while a keypress would be swallowed as a binding. The caller
    /// uses this to keep that press off the pad and off its own shortcuts.
    pub fn capturing(&self) -> bool {
        self.armed.is_some()
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Outcome {
        self.capture(ctx);

        let mut outcome = Outcome::Open;
        let mut open = true;
        egui::Window::new("Keyboard mapping")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size(egui::vec2(1000.0, 620.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Defaults").clicked() {
                        self.draft = KeyBindings::default();
                        self.armed = None;
                    }
                    ui.separator();
                    if ui.button("OK").clicked() {
                        outcome = Outcome::Accept(self.draft.clone());
                    }
                    if ui.button("Cancel").clicked() {
                        outcome = Outcome::Cancel;
                    }
                    ui.separator();
                    ui.label(match self.armed {
                        Some(i) => format!("press a key for {}  (Esc to cancel)", BUTTON_NAMES[i]),
                        None => "click a box, then press the key to bind".to_string(),
                    });
                });
                ui.separator();
                // Whatever is left of the window after the button row; the
                // diagram scales into it rather than forcing a scroll bar.
                let rect = ui.available_rect_before_wrap();
                ui.advance_cursor_after_rect(rect);
                self.diagram(ui, rect);
            });

        if !open && matches!(outcome, Outcome::Open) {
            // The window's own close button means the same as Cancel.
            outcome = Outcome::Cancel;
        }
        outcome
    }

    /// Take the first real keypress while armed. Repeats are ignored so a
    /// held key binds once, and Escape disarms instead of binding itself —
    /// otherwise the only way out of a misclick would be to bind Escape.
    fn capture(&mut self, ctx: &egui::Context) {
        let Some(idx) = self.armed else { return };
        let key = ctx.input_mut(|input| {
            let mut found = None;
            input.events.retain(|event| match event {
                egui::Event::Key { key, pressed: true, repeat: false, .. } if found.is_none() => {
                    found = Some(*key);
                    false
                }
                _ => true,
            });
            found
        });
        let Some(key) = key else { return };
        self.armed = None;
        if key != egui::Key::Escape {
            *self.draft.fields_mut()[idx] = key.name().to_string();
        }
    }

    fn diagram(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let view = View::fit(rect);
        let visuals = ui.visuals().clone();
        let ink = visuals.strong_text_color();
        let line = view.stroke(LINE, ink);
        let leader = view.stroke(LINE - 1.0, visuals.weak_text_color());

        let painter = ui.painter_at(rect);
        chassis(&painter, view, line, &visuals);

        // Leader lines before the buttons, so a button's fill covers the end
        // of the line rather than the line running across the button.
        for slot in &SLOTS {
            painter.add(egui::Shape::line(leader_points(view, slot), leader));
        }

        let paint = |behind: bool| {
            for slot in SLOTS.iter().filter(|s| s.behind == behind) {
                let armed = self.armed == Some(slot.idx);
                let fill = if armed { visuals.selection.bg_fill } else { visuals.extreme_bg_color };
                slot.art.paint(&painter, view, fill, line);
                slot.mark.paint(&painter, view, slot.art, ink);
            }
        };
        // The rings cross the shoulders and the face buttons sit inside the
        // right one, so the order is shoulders, rings, then everything else.
        paint(true);
        for x in [205.0, 695.0] {
            painter.circle_stroke(view.art(x, 230.0), view.len(120.0), line);
        }
        paint(false);
        for x in [315.0, 585.0] {
            painter.circle_stroke(view.art(x, 350.0), view.len(44.0), line);
        }

        let duplicates = self.duplicates();
        for slot in &SLOTS {
            self.label_box(ui, view, slot, duplicates, &visuals);
        }
    }

    /// A bitmask of the buttons whose key is bound to some other button as
    /// well. Both are flagged; neither is an error the dialog refuses, since
    /// two buttons on one key is occasionally what someone wants.
    fn duplicates(&self) -> u16 {
        let keys: Vec<&str> = self.draft.pairs().iter().map(|&(k, _)| k).collect();
        let mut mask = 0u16;
        for (i, a) in keys.iter().enumerate() {
            if keys.iter().enumerate().any(|(j, b)| j != i && a.eq_ignore_ascii_case(b)) {
                mask |= 1 << i;
            }
        }
        mask
    }

    fn label_box(
        &mut self,
        ui: &mut egui::Ui,
        view: View,
        slot: &Slot,
        duplicates: u16,
        visuals: &egui::Visuals,
    ) {
        let armed = self.armed == Some(slot.idx);
        let centre = view.at(slot.label.0, slot.label.1);
        let rect = egui::Rect::from_center_size(centre, BOX * view.scale);

        let text = if armed {
            "press a key".to_string()
        } else {
            self.draft.pairs()[slot.idx].0.to_string()
        };
        let stroke = if duplicates & (1 << slot.idx) != 0 {
            egui::Stroke::new(2.0, visuals.error_fg_color)
        } else {
            visuals.widgets.inactive.bg_stroke
        };
        let button = egui::Button::new(text).stroke(stroke).selected(armed);
        let response = ui
            .put(rect, button)
            .on_hover_text(format!("{} — click, then press a key", BUTTON_NAMES[slot.idx]));
        if response.clicked() {
            self.armed = if armed { None } else { Some(slot.idx) };
        }
    }
}

/// The two-segment leader line from a label box to its button.
fn leader_points(view: View, slot: &Slot) -> Vec<egui::Pos2> {
    let (bx, by) = slot.label;
    let anchor = (slot.anchor.0 + ORIGIN.x, slot.anchor.1 + ORIGIN.y);
    let half = BOX / 2.0;
    match slot.route {
        Route::Horizontal => {
            let edge = bx + half.x * (anchor.0 - bx).signum();
            let turn = slot.elbow.unwrap_or(anchor.0);
            vec![
                view.at(edge, by),
                view.at(turn, by),
                view.at(turn, anchor.1),
                view.at(anchor.0, anchor.1),
            ]
        }
        Route::Vertical => {
            let edge = by + half.y * (anchor.1 - by).signum();
            let turn = slot.elbow.unwrap_or(anchor.1);
            vec![
                view.at(bx, edge),
                view.at(bx, turn),
                view.at(anchor.0, turn),
                view.at(anchor.0, anchor.1),
            ]
        }
    }
}

/// Everything on the drawing that is not a bindable button: the body, the
/// two rings, and the parts this emulator has no binding for.
fn chassis(p: &egui::Painter, view: View, line: egui::Stroke, visuals: &egui::Visuals) {
    let mut body = Path::new(view, 185.0, 120.0);
    body.cubic(145.0, 120.0, 110.0, 142.0, 92.0, 176.0);
    body.line(55.0, 355.0);
    body.cubic(47.0, 392.0, 55.0, 430.0, 77.0, 458.0);
    body.cubic(95.0, 481.0, 121.0, 490.0, 144.0, 488.0);
    body.cubic(170.0, 485.0, 189.0, 465.0, 203.0, 437.0);
    body.line(243.0, 360.0);
    body.cubic(250.0, 348.0, 262.0, 340.0, 276.0, 340.0);
    body.line(624.0, 340.0);
    body.cubic(638.0, 340.0, 650.0, 348.0, 657.0, 360.0);
    body.line(697.0, 437.0);
    body.cubic(711.0, 465.0, 730.0, 485.0, 756.0, 488.0);
    body.cubic(779.0, 490.0, 805.0, 481.0, 823.0, 458.0);
    body.cubic(845.0, 430.0, 853.0, 392.0, 845.0, 355.0);
    body.line(808.0, 176.0);
    body.cubic(790.0, 142.0, 755.0, 120.0, 715.0, 120.0);
    body.line(562.0, 120.0);
    body.cubic(549.0, 120.0, 538.0, 114.0, 530.0, 105.0);
    body.cubic(507.0, 79.0, 468.0, 63.0, 450.0, 63.0);
    body.cubic(432.0, 63.0, 393.0, 79.0, 370.0, 105.0);
    body.cubic(362.0, 114.0, 351.0, 120.0, 338.0, 120.0);
    p.add(egui::Shape::closed_line(body.points, line));

    let mut bridge = Path::new(view, 287.0, 340.0);
    bridge.cubic(315.0, 314.0, 356.0, 302.0, 450.0, 302.0);
    bridge.cubic(544.0, 302.0, 585.0, 314.0, 613.0, 340.0);
    p.add(egui::Shape::line(bridge.points, line));

    // The analog toggle: drawn because the controller has one, not bound
    // because this frontend has no analog/digital switch to bind it to.
    Art::Rect { x: 427.0, y: 272.0, w: 46.0, h: 24.0, r: 6.0 }
        .paint(p, view, visuals.extreme_bg_color, line);

    let font = egui::FontId::proportional(view.len(18.0));
    for (x, text) in [(362.0, "SELECT"), (525.0, "START")] {
        p.text(view.art(x, 248.0), egui::Align2::CENTER_CENTER, text, font.clone(), line.color);
    }
    // A line lower than the other two, which it would otherwise crowd.
    let small = egui::FontId::proportional(view.len(15.0));
    p.text(view.art(450.0, 260.0), egui::Align2::CENTER_CENTER, "ANALOG", small, line.color);
}

impl Mark {
    fn paint(self, p: &egui::Painter, view: View, art: Art, ink: egui::Color32) {
        let (cx, cy) = match art {
            Art::Circle { x, y, .. } => (x, y),
            Art::Rect { x, y, w, h, .. } => (x + w / 2.0, y + h / 2.0),
            Art::Shoulder { x, y, w, h } => (x + w / 2.0, y + h * 0.40),
            Art::Tri(_) => return,
        };
        let thin = view.stroke(LINE - 0.5, ink);
        match self {
            Mark::None => {}
            Mark::Inside(text) => {
                let font = egui::FontId::proportional(view.len(26.0));
                p.text(view.art(cx, cy), egui::Align2::CENTER_CENTER, text, font, ink);
            }
            Mark::Triangle => {
                let pts = [(cx, cy - 20.0), (cx + 17.0, cy + 13.0), (cx - 17.0, cy + 13.0)];
                let pts: Vec<_> = pts.iter().map(|&(x, y)| view.art(x, y)).collect();
                p.add(egui::Shape::closed_line(pts, thin));
            }
            Mark::Circle => {
                p.circle_stroke(view.art(cx, cy), view.len(16.0), thin);
            }
            Mark::Cross => {
                p.line_segment([view.art(cx - 14.0, cy - 14.0), view.art(cx + 14.0, cy + 14.0)], thin);
                p.line_segment([view.art(cx + 14.0, cy - 14.0), view.art(cx - 14.0, cy + 14.0)], thin);
            }
            Mark::Square => {
                let rect =
                    egui::Rect::from_min_max(view.art(cx - 16.0, cy - 16.0), view.art(cx + 16.0, cy + 16.0));
                p.rect_stroke(rect, egui::CornerRadius::ZERO, thin, egui::StrokeKind::Middle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The diagram has to cover the pad exactly once, or a button would be
    /// unreachable from the dialog (or two boxes would edit one binding).
    #[test]
    fn every_button_has_one_slot() {
        let mut seen = 0u16;
        for slot in &SLOTS {
            assert!(seen & (1 << slot.idx) == 0, "slot {} listed twice", slot.idx);
            seen |= 1 << slot.idx;
        }
        assert_eq!(seen, u16::MAX);
    }

    /// Capture stores `Key::name()`, and startup resolves the stored name
    /// with `Key::from_name()`. Anything bindable has to survive that trip.
    #[test]
    fn key_names_round_trip() {
        for &key in egui::Key::ALL {
            assert_eq!(egui::Key::from_name(key.name()), Some(key), "{key:?}");
        }
    }

    /// `fields_mut` is indexed with `pairs()` indices, so the two have to
    /// walk the bindings in the same order.
    #[test]
    fn field_order_matches_pair_order() {
        let mut keys = KeyBindings::default();
        for i in 0..16 {
            *keys.fields_mut()[i] = format!("probe{i}");
        }
        for (i, (name, _)) in keys.pairs().iter().enumerate() {
            assert_eq!(*name, format!("probe{i}"));
        }
    }
}
