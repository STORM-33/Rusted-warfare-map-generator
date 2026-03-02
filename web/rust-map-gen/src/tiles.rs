pub const TILE_ID_OFFSET: i32 = 201;
pub const DECORATION_FREQUENCY: f64 = 0.05;

pub const WATER_SAND: [i32; 14] = [31, 34, 6, 7, 8, 33, 35, 60, 61, 62, 87, 88, 114, 115];
pub const SAND_GRASS: [i32; 14] = [34, 37, 9, 10, 11, 36, 38, 63, 64, 65, 90, 91, 117, 118];
pub const GRASS_SOIL: [i32; 14] = [37, 40, 12, 13, 14, 39, 41, 66, 67, 68, 93, 94, 120, 121];
pub const SOIL_SWAMP: [i32; 14] = [40, 43, 15, 16, 17, 42, 44, 69, 70, 71, 96, 97, 123, 124];
pub const SWAMP_STONE: [i32; 14] = [43, 46, 18, 19, 20, 45, 47, 72, 73, 74, 99, 100, 126, 127];
pub const STONE_SNOW: [i32; 14] = [46, 49, 21, 22, 23, 48, 50, 75, 76, 77, 102, 103, 129, 130];
pub const SNOW_ICE: [i32; 14] = [49, 52, 24, 25, 26, 51, 53, 78, 79, 80, 105, 106, 132, 133];
pub const DEEP_WATER_WATER: [i32; 14] = [28, 31, 3, 4, 5, 30, 32, 57, 58, 59, 84, 85, 111, 112];
pub const OCEAN_DEEP_WATER: [i32; 14] = [83, 28, 0, 1, 2, 27, 29, 54, 55, 56, 81, 82, 108, 109];
pub const LARGE_ROCK_TILE_SET: [i32; 14] = [-1, 4, 0, 1, 2, 3, 5, 6, 7, 8, 13, 12, 10, 9];

pub const TERRAIN_LEVELS: [(i32, [i32; 14]); 9] = [
    (-1, OCEAN_DEEP_WATER),
    (0, DEEP_WATER_WATER),
    (1, WATER_SAND),
    (2, SAND_GRASS),
    (3, GRASS_SOIL),
    (4, SOIL_SWAMP),
    (5, SWAMP_STONE),
    (6, STONE_SNOW),
    (7, SNOW_ICE),
];

pub const DECORATION_TILES_LEVEL_1: [i32; 3] = [86, 89, 116];
pub const DECORATION_TILES_LEVEL_2: [i32; 2] = [110, 119];
pub const DECORATION_TILES_LEVEL_3: [i32; 2] = [95, 122];
pub const DECORATION_TILES_LEVEL_4: [i32; 2] = [98, 125];
pub const DECORATION_TILES_LEVEL_5: [i32; 2] = [101, 128];
pub const DECORATION_TILES_LEVEL_6: [i32; 2] = [104, 131];
pub const DECORATION_TILES_LEVEL_7: [i32; 2] = [107, 134];
