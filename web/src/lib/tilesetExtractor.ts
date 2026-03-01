export interface ExtractedTileset {
  name: string;
  firstGid: number;
  columns: number;
  tileCount: number;
  tileWidth: number;
  tileHeight: number;
  image: ImageBitmap;
  centerLut: Uint8ClampedArray;
}

export interface ExtractedTilesets {
  ground?: ExtractedTileset;
  wall?: ExtractedTileset;
  items?: ExtractedTileset;
  units?: ExtractedTileset;
}

const decodeBase64 = (value: string) => {
  const normalized = value.replace(/\s+/g, "");
  const binary = atob(normalized);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
};

const buildCenterLut = (
  imageData: ImageData,
  tileWidth: number,
  tileHeight: number,
  columns: number,
  tileCount: number,
) => {
  const lut = new Uint8ClampedArray(tileCount * 4);
  for (let tileId = 0; tileId < tileCount; tileId += 1) {
    const tileRow = Math.floor(tileId / columns);
    const tileCol = tileId % columns;
    const cx = tileCol * tileWidth + Math.floor(tileWidth / 2);
    const cy = tileRow * tileHeight + Math.floor(tileHeight / 2);
    const src = (cy * imageData.width + cx) * 4;
    const dst = tileId * 4;
    lut[dst] = imageData.data[src];
    lut[dst + 1] = imageData.data[src + 1];
    lut[dst + 2] = imageData.data[src + 2];
    lut[dst + 3] = imageData.data[src + 3];
  }
  return lut;
};

const extractTileset = async (tilesetElement: Element) => {
  const name = tilesetElement.getAttribute("name") ?? "";
  const firstGid = Number(tilesetElement.getAttribute("firstgid") ?? "1");
  const columns = Number(tilesetElement.getAttribute("columns") ?? "1");
  const tileCount = Number(tilesetElement.getAttribute("tilecount") ?? "0");
  const tileWidth = Number(tilesetElement.getAttribute("tilewidth") ?? "20");
  const tileHeight = Number(tilesetElement.getAttribute("tileheight") ?? "20");

  const properties = Array.from(tilesetElement.querySelectorAll("property"));
  const embeddedPng = properties.find(
    (property) => property.getAttribute("name") === "embedded_png",
  )?.textContent;

  if (!embeddedPng) {
    return undefined;
  }

  const bytes = decodeBase64(embeddedPng);
  const blob = new Blob([bytes], { type: "image/png" });
  const image = await createImageBitmap(blob);

  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    return undefined;
  }
  context.drawImage(image, 0, 0);
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const centerLut = buildCenterLut(
    imageData,
    tileWidth,
    tileHeight,
    columns,
    tileCount,
  );

  return {
    name,
    firstGid,
    columns,
    tileCount,
    tileWidth,
    tileHeight,
    image,
    centerLut,
  } as ExtractedTileset;
};

export async function extractTilesetsFromTmx(tmxXml: string) {
  const parser = new DOMParser();
  const documentNode = parser.parseFromString(tmxXml, "application/xml");
  const tilesetElements = Array.from(documentNode.querySelectorAll("tileset"));
  const extracted = await Promise.all(tilesetElements.map(extractTileset));

  const result: ExtractedTilesets = {};
  for (const tileset of extracted) {
    if (!tileset) {
      continue;
    }
    if (tileset.name === "AutoLight") {
      result.ground = tileset;
    } else if (tileset.name === "large-rock") {
      result.wall = tileset;
    } else if (tileset.name === "export_items") {
      result.items = tileset;
    } else if (tileset.name === "50pCommandCenter") {
      result.units = tileset;
    }
  }
  return result;
}

export async function loadTilesetsFromBlueprint(blueprintUrl: string) {
  const response = await fetch(blueprintUrl);
  if (!response.ok) {
    throw new Error(`Failed to load blueprint ${blueprintUrl}`);
  }
  const xml = await response.text();
  return extractTilesetsFromTmx(xml);
}
