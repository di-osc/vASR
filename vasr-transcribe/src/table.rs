use tabled::{
    builder::Builder,
    settings::{Alignment, Panel, Style},
};

pub(crate) fn render_table<const N: usize>(
    header: [&'static str; N],
    rows: impl IntoIterator<Item = Vec<String>>,
) -> String {
    render_table_inner(None, header, rows)
}

pub(crate) fn render_titled_table<const N: usize>(
    title: String,
    header: [&'static str; N],
    rows: impl IntoIterator<Item = Vec<String>>,
) -> String {
    render_table_inner(Some(title), header, rows)
}

fn render_table_inner<const N: usize>(
    title: Option<String>,
    header: [&'static str; N],
    rows: impl IntoIterator<Item = Vec<String>>,
) -> String {
    let mut builder = Builder::default();
    builder.push_record(header);
    for row in rows {
        builder.push_record(row);
    }

    let mut table = builder.build();
    table.with(Style::ascii()).with(Alignment::center());
    if let Some(title) = title {
        table.with(Panel::header(title));
    }
    table.to_string()
}
