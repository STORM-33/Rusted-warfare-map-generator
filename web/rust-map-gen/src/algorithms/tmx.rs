use std::io::Write;



use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;

use base64::Engine;

use flate2::write::GzEncoder;

use flate2::Compression;

use xmltree::{Element, XMLNode};



use crate::state::{Matrix, WizardState};

use crate::tiles::TILE_ID_OFFSET;



pub(crate) fn write_tmx(state: &WizardState, blueprint_xml: &str) -> Result<Vec<u8>, String> {
    let id_matrix = state
        .id_matrix
        .as_ref()
        .ok_or_else(|| "Missing id_matrix for TMX export".to_string())?;
    let height_map = state
        .height_map
        .as_ref()
        .ok_or_else(|| "Missing height_map for TMX export".to_string())?;
    let items_matrix = state
        .items_matrix
        .clone()
        .unwrap_or_else(|| Matrix::zeros(height_map.rows, height_map.cols));
    let units_matrix = state
        .units_matrix
        .clone()
        .unwrap_or_else(|| Matrix::zeros(height_map.rows, height_map.cols));

    let h = id_matrix.rows;
    let w = id_matrix.cols;

    let base64_items = encode_tmx_layer(&items_matrix, 0)?;
    let base64_units = encode_tmx_layer(&units_matrix, 0)?;
    let base64_ground = encode_tmx_layer(id_matrix, TILE_ID_OFFSET)?;

    let mut root = Element::parse(blueprint_xml.as_bytes())
        .map_err(|err| format!("Failed to parse blueprint XML: {err}"))?;

    root.attributes.insert("width".to_string(), w.to_string());
    root.attributes.insert("height".to_string(), h.to_string());

    update_layer_data(&mut root, "Ground", &base64_ground, w, h);
    update_layer_data(&mut root, "Items", &base64_items, w, h);
    update_layer_data(&mut root, "Units", &base64_units, w, h);
    ensure_triggers_map_info(&mut root, w, h);

    let mut out = Vec::new();
    root.write_with_config(
        &mut out,
        xmltree::EmitterConfig::new()
            .perform_indent(false)
            .write_document_declaration(true),
    )
    .map_err(|err| format!("Failed to serialize TMX XML: {err}"))?;
    Ok(out)
}



fn update_layer_data(root: &mut Element, layer_name: &str, data: &str, w: usize, h: usize) {
    for child in &mut root.children {
        let XMLNode::Element(layer) = child else {
            continue;
        };
        if layer.name != "layer" {
            continue;
        }
        if layer.attributes.get("name").map(String::as_str) != Some(layer_name) {
            continue;
        }
        layer.attributes.insert("width".to_string(), w.to_string());
        layer.attributes.insert("height".to_string(), h.to_string());

        let mut data_idx = None;
        for (idx, node) in layer.children.iter().enumerate() {
            if let XMLNode::Element(elem) = node {
                if elem.name == "data" {
                    data_idx = Some(idx);
                    break;
                }
            }
        }
        if let Some(idx) = data_idx {
            if let Some(XMLNode::Element(data_elem)) = layer.children.get_mut(idx) {
                data_elem.children.clear();
                data_elem.children.push(XMLNode::Text(data.to_string()));
            }
        }
    }
}



fn ensure_triggers_map_info(root: &mut Element, w: usize, h: usize) {
    let triggers_idx = find_child_by_attr_index(root, "objectgroup", "name", "Triggers");
    let triggers_index = if let Some(idx) = triggers_idx {
        idx
    } else {
        let mut triggers = Element::new("objectgroup");
        triggers
            .attributes
            .insert("name".to_string(), "Triggers".to_string());
        root.children.push(XMLNode::Element(triggers));
        root.children.len() - 1
    };

    let Some(XMLNode::Element(triggers)) = root.children.get_mut(triggers_index) else {
        return;
    };

    let map_info_idx = find_child_by_attr_index(triggers, "object", "name", "map_info");
    if map_info_idx.is_none() {
        let mut map_info = Element::new("object");
        map_info
            .attributes
            .insert("id".to_string(), "1".to_string());
        map_info
            .attributes
            .insert("name".to_string(), "map_info".to_string());
        map_info.attributes.insert("x".to_string(), "0".to_string());
        map_info.attributes.insert("y".to_string(), "0".to_string());
        map_info
            .attributes
            .insert("width".to_string(), (w * 20).to_string());
        map_info
            .attributes
            .insert("height".to_string(), (h * 20).to_string());

        let mut properties = Element::new("properties");
        let mut type_prop = Element::new("property");
        type_prop
            .attributes
            .insert("name".to_string(), "type".to_string());
        type_prop
            .attributes
            .insert("value".to_string(), "skirmish".to_string());

        let mut intro_prop = Element::new("property");
        intro_prop
            .attributes
            .insert("name".to_string(), "introText".to_string());
        intro_prop.attributes.insert(
            "value".to_string(),
            "Map generated with Rusted Warfare Map Generator by STORM\\nhttps://web-two-delta-61.vercel.app"
                .to_string(),
        );
        properties.children.push(XMLNode::Element(type_prop));
        properties.children.push(XMLNode::Element(intro_prop));
        map_info.children.push(XMLNode::Element(properties));
        triggers.children.push(XMLNode::Element(map_info));
    }
}



fn find_child_by_attr_index(
    element: &Element,
    child_name: &str,
    attr_name: &str,
    attr_value: &str,
) -> Option<usize> {
    for (idx, node) in element.children.iter().enumerate() {
        let XMLNode::Element(child) = node else {
            continue;
        };
        if child.name == child_name
            && child.attributes.get(attr_name).map(String::as_str) == Some(attr_value)
        {
            return Some(idx);
        }
    }
    None
}



fn encode_tmx_layer(matrix: &Matrix, offset: i32) -> Result<String, String> {
    let mut bytes = Vec::with_capacity(matrix.data.len() * 4);
    for value in &matrix.data {
        let encoded = (*value + offset) as u32;
        bytes.extend(encoded.to_le_bytes());
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&bytes)
        .map_err(|err| format!("Failed to gzip TMX layer: {err}"))?;
    let compressed = encoder
        .finish()
        .map_err(|err| format!("Failed to finalize gzip stream: {err}"))?;
    Ok(BASE64_STANDARD.encode(compressed))
}
