#include <M5Capsule.h>
#include <NeoPixelBus.h>

#define NUM_PORTS 5
#define TRIANGLES_PER_PORT 16
#define LEDS_PER_TRIANGLE 36
#define LEDS_PER_PORT (TRIANGLES_PER_PORT * LEDS_PER_TRIANGLE)
#define TOTAL_LEDS (NUM_PORTS * LEDS_PER_PORT)
const int ledPins[NUM_PORTS] = { 1, 3, 5, 7, 9 };
NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod> *strips[NUM_PORTS];
const uint8_t BRIGHTNESS = 1;
RgbColor ledBuffer[TOTAL_LEDS];

// --- 物理LEDインデックス（画像の配線順に必ず修正してください） ---
const uint16_t triangle_led_physical_index[5][16][36] = {
    // 切片1
    {
        {   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36 },
        {  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72 },
        {  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108 },
        { 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144 },
        { 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180 },
        { 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216 },
        { 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252 },
        { 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288 },
        { 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324 },
        { 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360 },
        { 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396 },
        { 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432 },
        { 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468 },
        { 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504 },
        { 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540 },
        { 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576 }
    },
    // 切片2～5も同様に記載（省略せず、各LED番号を36個ずつ増やしつつ記入）
    // ...
};

// --- 三角形内LEDローカル座標（例：均等配置。実際の物理配置に合わせて修正） ---
const float triangle_local_coords[36][2] = {
    {0.5, 1.0}, {0.42, 0.92}, {0.58, 0.92}, {0.34, 0.84}, {0.5, 0.84}, {0.66, 0.84},
    {0.26, 0.76}, {0.42, 0.76}, {0.58, 0.76}, {0.74, 0.76}, {0.18, 0.68}, {0.34, 0.68},
    {0.5, 0.68}, {0.66, 0.68}, {0.82, 0.68}, {0.10, 0.60}, {0.26, 0.60}, {0.42, 0.60},
    {0.58, 0.60}, {0.74, 0.60}, {0.90, 0.60}, {0.02, 0.52}, {0.18, 0.52}, {0.34, 0.52},
    {0.50, 0.52}, {0.66, 0.52}, {0.82, 0.52}, {0.98, 0.52}, {0.10, 0.36}, {0.26, 0.36},
    {0.42, 0.36}, {0.58, 0.36}, {0.74, 0.36}, {0.90, 0.36}, {0.18, 0.20}, {0.34, 0.20}
};

const int TRIANGLE_GRID_W = 5;
const int TRIANGLE_GRID_H = 16;
const int IMAGE_W = 180;
const int IMAGE_H = 32;

void get_triangle_grid_pos(int segment, int tri, int &grid_x, int &grid_y) {
    grid_x = segment;
    grid_y = tri;
}

void get_image_pixel_for_led(int segment, int tri, int led, int &img_x, int &img_y) {
    int grid_x, grid_y;
    get_triangle_grid_pos(segment, tri, grid_x, grid_y);
    float u = triangle_local_coords[led][0];
    float v = triangle_local_coords[led][1];
    int tri_img_w = IMAGE_W / TRIANGLE_GRID_W;
    int tri_img_h = IMAGE_H / TRIANGLE_GRID_H;
    img_x = grid_x * tri_img_w + int(u * (tri_img_w-1));
    img_y = grid_y * tri_img_h + int(v * (tri_img_h-1));
}

void drawImageToBuffer(const uint8_t *img, RgbColor *ledBuffer) {
    for (int seg = 0; seg < 5; ++seg) {
        for (int tri = 0; tri < 16; ++tri) {
            for (int led = 0; led < 36; ++led) {
                int img_x, img_y;
                get_image_pixel_for_led(seg, tri, led, img_x, img_y);
                int img_idx = (img_y * IMAGE_W + img_x) * 3;
                uint8_t r = img[img_idx + 0];
                uint8_t g = img[img_idx + 1];
                uint8_t b = img[img_idx + 2];
                r = (r * BRIGHTNESS) / 255;
                g = (g * BRIGHTNESS) / 255;
                b = (b * BRIGHTNESS) / 255;
                int led_idx = triangle_led_physical_index[seg][tri][led] - 1;
                ledBuffer[led_idx] = RgbColor(r, g, b);
            }
        }
    }
}

void showBuffer() {
    for (int p = 0; p < NUM_PORTS; ++p) {
        for (int i = 0; i < LEDS_PER_PORT; ++i) {
            strips[p]->SetPixelColor(i, ledBuffer[p * LEDS_PER_PORT + i]);
        }
        strips[p]->Show();
    }
}

class InteractionManager {
public:
    void begin() {}
    void update() {}
};
InteractionManager interaction;

void setup() {
    M5.begin();
    for (int p = 0; p < NUM_PORTS; ++p) {
        strips[p] = new NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod>(LEDS_PER_PORT, ledPins[p]);
        strips[p]->Begin();
        strips[p]->Show();
    }
    interaction.begin();
}

void loop() {
    // 動的なレインボー画像データを生成
    static uint8_t dummy_img[IMAGE_W * IMAGE_H * 3];
    static uint32_t frame = 0;
    frame++;
    for (int y = 0; y < IMAGE_H; ++y) {
        for (int x = 0; x < IMAGE_W; ++x) {
            float hue = fmodf((float)x / IMAGE_W * 360.0f + frame * 2, 360.0f); // 横方向に虹色＋時間で流れる
            float s = 1.0f, v = 1.0f;
            float c = v * s;
            float xh = c * (1 - fabs(fmodf(hue / 60.0f, 2) - 1));
            float m = v - c;
            float r1, g1, b1;
            if (hue < 60)      { r1 = c; g1 = xh; b1 = 0; }
            else if (hue < 120) { r1 = xh; g1 = c; b1 = 0; }
            else if (hue < 180) { r1 = 0; g1 = c; b1 = xh; }
            else if (hue < 240) { r1 = 0; g1 = xh; b1 = c; }
            else if (hue < 300) { r1 = xh; g1 = 0; b1 = c; }
            else                { r1 = c; g1 = 0; b1 = xh; }
            uint8_t r = (uint8_t)((r1 + m) * 255);
            uint8_t g = (uint8_t)((g1 + m) * 255);
            uint8_t b = (uint8_t)((b1 + m) * 255);
            int idx = (y * IMAGE_W + x) * 3;
            dummy_img[idx + 0] = r;
            dummy_img[idx + 1] = g;
            dummy_img[idx + 2] = b;
        }
    }
    drawImageToBuffer(dummy_img, ledBuffer);
    showBuffer();
    interaction.update();
    delay(16); // 60fps
}
