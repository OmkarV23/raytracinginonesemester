// include/scene.h
#pragma once

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "vec3.h"
#include "camera.h"
#include "material.h"
#include "medium.h"

struct SceneSettings {
    int max_depth = 1;
    int spp = 1;
    bool diffuse_bounce = true;
};

struct Light {
    Vec3 position     = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 color        = make_vec3(1.0f, 1.0f, 1.0f);
    int  intensity    = 1;

    int  type         = 0;                              // 0 = point, 1 = area
    Vec3 edge_u       = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 edge_v       = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 emission     = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 light_normal = make_vec3(0.0f, 1.0f, 0.0f);
    float area        = 1.0f;
};

struct SceneObject {
    std::string name;
    std::string type;
    std::string path;
    Vec3 position = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 rotation = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 scale    = make_vec3(1.0f, 1.0f, 1.0f);
    Material material;

    // ---- NEW: per-object participating medium ----
    HomogeneousMedium medium;

    // ---- NEW: texture overrides from scene JSON ----
    std::string diffuse_tex_path;   // override diffuse texture (or empty = use MTL)
    std::string normal_tex_path;    // override normal map (or empty = use MTL)
    bool        use_mtl = true;     // if true, auto-load mtllib from OBJ
    
    // ---- NEW: volume-only flag (no geometry, just medium in space) ----
    bool        is_volume = false;  // if true, register medium but skip OBJ loading
};

// Volume Region: AABB-based spatial volume with medium
struct VolumeRegion {
    Vec3 min_bounds;
    Vec3 max_bounds;
    HomogeneousMedium medium;
    
    // Check if point is inside this volume
    HYBRID_FUNC inline bool contains(const Vec3& p) const {
        return p.x >= min_bounds.x && p.x <= max_bounds.x &&
               p.y >= min_bounds.y && p.y <= max_bounds.y &&
               p.z >= min_bounds.z && p.z <= max_bounds.z;
    }
};

struct Scene {
    SceneSettings settings;
    Camera camera;
    Vec3 miss_color = make_vec3(0.0f, 0.0f, 0.0f);
    std::vector<Light> lights;
    std::vector<SceneObject> objects;
    std::vector<VolumeRegion> volumes;  // NEW: spatial volume regions
};

// ============================================================
// JSON parser (unchanged, just the SceneIO namespace)
// ============================================================
namespace SceneIO {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;
};

struct JsonParser {
    const std::string* s = nullptr;
    size_t i = 0;
    std::string err;

    void skip_ws() {
        while (i < s->size() && std::isspace(static_cast<unsigned char>((*s)[i]))) ++i;
    }

    bool parse_value(JsonValue& out) {
        skip_ws();
        if (i >= s->size()) return fail("Unexpected end of input");
        char c = (*s)[i];
        if (c == '{') return parse_object(out);
        if (c == '[') return parse_array(out);
        if (c == '"') return parse_string(out);
        if (c == 't' || c == 'f') return parse_bool(out);
        if (c == 'n') return parse_null(out);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out);
        return fail("Unexpected character");
    }

    bool parse_object(JsonValue& out) {
        out.type = JsonValue::Type::Object;
        out.obj.clear();
        ++i;
        skip_ws();
        if (i < s->size() && (*s)[i] == '}') { ++i; return true; }
        while (i < s->size()) {
            JsonValue key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (i >= s->size() || (*s)[i] != ':') return fail("Expected ':'");
            ++i;
            JsonValue val;
            if (!parse_value(val)) return false;
            out.obj.emplace_back(key.str, std::move(val));
            skip_ws();
            if (i < s->size() && (*s)[i] == ',') { ++i; skip_ws(); continue; }
            if (i < s->size() && (*s)[i] == '}') { ++i; return true; }
            return fail("Expected ',' or '}'");
        }
        return fail("Unterminated object");
    }

    bool parse_array(JsonValue& out) {
        out.type = JsonValue::Type::Array;
        out.arr.clear();
        ++i;
        skip_ws();
        if (i < s->size() && (*s)[i] == ']') { ++i; return true; }
        while (i < s->size()) {
            JsonValue val;
            if (!parse_value(val)) return false;
            out.arr.push_back(std::move(val));
            skip_ws();
            if (i < s->size() && (*s)[i] == ',') { ++i; skip_ws(); continue; }
            if (i < s->size() && (*s)[i] == ']') { ++i; return true; }
            return fail("Expected ',' or ']'");
        }
        return fail("Unterminated array");
    }

    bool parse_string(JsonValue& out) {
        out.type = JsonValue::Type::String;
        out.str.clear();
        if ((*s)[i] != '"') return fail("Expected '\"'");
        ++i;
        while (i < s->size()) {
            char c = (*s)[i++];
            if (c == '"') return true;
            if (c == '\\') {
                if (i >= s->size()) return fail("Bad escape");
                char e = (*s)[i++];
                switch (e) {
                    case '"':  out.str.push_back('"');  break;
                    case '\\': out.str.push_back('\\'); break;
                    case '/':  out.str.push_back('/');  break;
                    case 'b':  out.str.push_back('\b'); break;
                    case 'f':  out.str.push_back('\f'); break;
                    case 'n':  out.str.push_back('\n'); break;
                    case 'r':  out.str.push_back('\r'); break;
                    case 't':  out.str.push_back('\t'); break;
                    case 'u': {
                        if (i + 4 > s->size()) return fail("Bad unicode escape");
                        unsigned code = 0;
                        for (int k = 0; k < 4; ++k) {
                            char h = (*s)[i++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= (h - '0');
                            else if (h >= 'a' && h <= 'f') code |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= (h - 'A' + 10);
                            else return fail("Bad unicode escape");
                        }
                        if (code <= 0x7F) out.str.push_back(static_cast<char>(code));
                        else out.str.push_back('?');
                    } break;
                    default: return fail("Bad escape");
                }
            } else {
                out.str.push_back(c);
            }
        }
        return fail("Unterminated string");
    }

    bool parse_number(JsonValue& out) {
        out.type = JsonValue::Type::Number;
        const char* start = s->c_str() + i;
        char* end = nullptr;
        out.num = std::strtod(start, &end);
        if (end == start) return fail("Bad number");
        i = static_cast<size_t>(end - s->c_str());
        return true;
    }

    bool parse_bool(JsonValue& out) {
        if (s->compare(i, 4, "true") == 0) { out.type = JsonValue::Type::Bool; out.b = true; i += 4; return true; }
        if (s->compare(i, 5, "false") == 0) { out.type = JsonValue::Type::Bool; out.b = false; i += 5; return true; }
        return fail("Bad boolean");
    }

    bool parse_null(JsonValue& out) {
        if (s->compare(i, 4, "null") == 0) { out.type = JsonValue::Type::Null; i += 4; return true; }
        return fail("Bad null");
    }

    bool fail(const std::string& msg) { err = msg; return false; }
};

inline bool parse_json(const std::string& text, JsonValue& out, std::string* err) {
    JsonParser p;
    p.s = &text;
    if (!p.parse_value(out)) { if (err) *err = p.err; return false; }
    p.skip_ws();
    if (p.i != text.size()) { if (err) *err = "Trailing characters"; return false; }
    return true;
}

inline bool json_get(const JsonValue& obj, const char* key, const JsonValue** out) {
    if (obj.type != JsonValue::Type::Object) return false;
    for (auto& kv : obj.obj) {
        if (kv.first == key) { *out = &kv.second; return true; }
    }
    return false;
}

inline bool json_as_vec3(const JsonValue& v, Vec3& out) {
    if (v.type != JsonValue::Type::Array || v.arr.size() != 3) return false;
    if (v.arr[0].type != JsonValue::Type::Number) return false;
    if (v.arr[1].type != JsonValue::Type::Number) return false;
    if (v.arr[2].type != JsonValue::Type::Number) return false;
    out = make_vec3(
        static_cast<float>(v.arr[0].num),
        static_cast<float>(v.arr[1].num),
        static_cast<float>(v.arr[2].num));
    return true;
}

// Helper: parse a "medium" JSON object into a HomogeneousMedium
inline void parse_medium(const JsonValue& medObj, HomogeneousMedium& med) {
    const JsonValue* v = nullptr;
    if (json_get(medObj, "sigma_a", &v) && v->type == JsonValue::Type::Number)
        med.sigma_a = static_cast<float>(v->num);
    if (json_get(medObj, "sigma_s", &v) && v->type == JsonValue::Type::Number)
        med.sigma_s = static_cast<float>(v->num);
    if (json_get(medObj, "g", &v) && v->type == JsonValue::Type::Number)
        med.g = static_cast<float>(v->num);
    med.compute_sigma_t();
    med.enabled = (med.sigma_t > 0.0f);
}

inline bool parse_scene(const JsonValue& root, Scene& scene, std::string* err) {
    if (root.type != JsonValue::Type::Object) {
        if (err) *err = "Root is not an object";
        return false;
    }

    const JsonValue* settings = nullptr;
    if (json_get(root, "settings", &settings)) {
        const JsonValue* max_bounces = nullptr;
        if (json_get(*settings, "max_bounces", &max_bounces) &&
            max_bounces->type == JsonValue::Type::Number) {
            scene.settings.max_depth = static_cast<int>(max_bounces->num);
        }
        const JsonValue* spp_val = nullptr;
        if (json_get(*settings, "spp", &spp_val) &&
            spp_val->type == JsonValue::Type::Number) {
            scene.settings.spp = static_cast<int>(spp_val->num);
            if (scene.settings.spp < 1) scene.settings.spp = 1;
        }
        const JsonValue* diffuse_bounce = nullptr;
        if (json_get(*settings, "diffuse_bounce", &diffuse_bounce) &&
            diffuse_bounce->type == JsonValue::Type::Bool) {
            scene.settings.diffuse_bounce = diffuse_bounce->b;
        }
    }

    const JsonValue* miss_color = nullptr;
    if (json_get(root, "miss_color", &miss_color)) {
        json_as_vec3(*miss_color, scene.miss_color);
    }

    const JsonValue* camera = nullptr;
    if (json_get(root, "camera", &camera)) {
        Vec3 cam_pos = scene.camera.get_center();
        Vec3 cam_look_at = scene.camera.get_look_at();
        Vec3 cam_up = scene.camera.get_up_vector();
        double focal_length_mm = scene.camera.get_focal_length_mm();
        double sensor_height_mm = scene.camera.get_sensor_height_mm();
        int pixel_width = scene.camera.pixel_width;
        int pixel_height = scene.camera.pixel_height;

        const JsonValue* v = nullptr;
        if (json_get(*camera, "focal_length_mm", &v) && v->type == JsonValue::Type::Number)
            focal_length_mm = static_cast<double>(v->num);
        if (json_get(*camera, "sensor_height_mm", &v) && v->type == JsonValue::Type::Number)
            sensor_height_mm = static_cast<double>(v->num);
        if (json_get(*camera, "pixel_width", &v) && v->type == JsonValue::Type::Number)
            pixel_width = static_cast<int>(v->num);
        if (json_get(*camera, "pixel_height", &v) && v->type == JsonValue::Type::Number)
            pixel_height = static_cast<int>(v->num);
        if (json_get(*camera, "position", &v)) json_as_vec3(*v, cam_pos);
        if (json_get(*camera, "look_at", &v)) json_as_vec3(*v, cam_look_at);
        if (json_get(*camera, "up", &v)) json_as_vec3(*v, cam_up);

        scene.camera = Camera(
            cam_pos, cam_look_at, cam_up,
            focal_length_mm, sensor_height_mm,
            pixel_width, pixel_height
        );
    }

    auto parse_one_light = [](const JsonValue& item) -> Light {
        Light lc;
        const JsonValue* v = nullptr;
        if (json_get(item, "position", &v)) json_as_vec3(*v, lc.position);
        if (json_get(item, "color",    &v)) json_as_vec3(*v, lc.color);
        if (json_get(item, "intensity", &v) && v->type == JsonValue::Type::Number)
            lc.intensity = static_cast<int>(v->num);

        const JsonValue* ltype = nullptr;
        if (json_get(item, "light_type", &ltype) &&
            ltype->type == JsonValue::Type::String &&
            ltype->str == "area") {

            lc.type = 1;
            if (json_get(item, "emission", &v)) json_as_vec3(*v, lc.emission);

            const JsonValue* eu = nullptr;
            const JsonValue* ev = nullptr;
            if (json_get(item, "edge_u", &eu) && json_get(item, "edge_v", &ev)) {
                json_as_vec3(*eu, lc.edge_u);
                json_as_vec3(*ev, lc.edge_v);
            } else {
                Vec3 N = make_vec3(0.0f, -1.0f, 0.0f);
                if (json_get(item, "normal", &v)) json_as_vec3(*v, N);
                float w = 1.0f, h = 1.0f;
                if (json_get(item, "width",  &v) && v->type == JsonValue::Type::Number)
                    w = static_cast<float>(v->num);
                if (json_get(item, "height", &v) && v->type == JsonValue::Type::Number)
                    h = static_cast<float>(v->num);

                Vec3 up = (fabsf(N.z) < 0.9f) ? make_vec3(0,0,1) : make_vec3(1,0,0);
                Vec3 cr = cross(up, N);
                float crLen = sqrtf(dot(cr, cr));
                Vec3 tangent   = (crLen > 1e-10f) ? cr * (1.0f / crLen) : make_vec3(1,0,0);
                Vec3 bitangent = cross(N, tangent);
                lc.edge_u = tangent   * (w * 0.5f);
                lc.edge_v = bitangent * (h * 0.5f);
            }

            Vec3 cr = cross(lc.edge_u, lc.edge_v);
            float crLen = sqrtf(dot(cr, cr));
            if (crLen > 1e-10f) {
                lc.light_normal = cr * (1.0f / crLen);
                lc.area = crLen * 4.0f;
            }
        }
        return lc;
    };

    scene.lights.clear();
    const JsonValue* lights = nullptr;
    if (json_get(root, "lights", &lights) && lights->type == JsonValue::Type::Array) {
        for (const auto& item : lights->arr) {
            if (item.type != JsonValue::Type::Object) continue;
            scene.lights.push_back(parse_one_light(item));
        }
    }
    if (scene.lights.empty()) {
        const JsonValue* light = nullptr;
        if (json_get(root, "light", &light) && light->type == JsonValue::Type::Object) {
            scene.lights.push_back(parse_one_light(*light));
        }
    }

    const JsonValue* scene_arr = nullptr;
    if (!json_get(root, "scene", &scene_arr) || scene_arr->type != JsonValue::Type::Array) {
        if (err) *err = "Missing 'scene' array";
        return false;
    }

    scene.objects.clear();
    for (const auto& item : scene_arr->arr) {
        if (item.type != JsonValue::Type::Object) continue;
        SceneObject obj;
        const JsonValue* v = nullptr;
        if (json_get(item, "name", &v) && v->type == JsonValue::Type::String) obj.name = v->str;
        if (json_get(item, "type", &v) && v->type == JsonValue::Type::String) obj.type = v->str;
        if (json_get(item, "path", &v) && v->type == JsonValue::Type::String) obj.path = v->str;

        const JsonValue* transform = nullptr;
        if (json_get(item, "transform", &transform) && transform->type == JsonValue::Type::Object) {
            if (json_get(*transform, "position", &v)) json_as_vec3(*v, obj.position);
            if (json_get(*transform, "rotation", &v)) json_as_vec3(*v, obj.rotation);
            if (json_get(*transform, "scale", &v))    json_as_vec3(*v, obj.scale);
        }

        const JsonValue* material = nullptr;
        if (json_get(item, "material", &material) && material->type == JsonValue::Type::Object) {
            if (json_get(*material, "albedo", &v))         json_as_vec3(*v, obj.material.albedo);
            if (json_get(*material, "specular_color", &v)) json_as_vec3(*v, obj.material.specularColor);
            if (json_get(*material, "emission", &v))       json_as_vec3(*v, obj.material.emission);
            if (json_get(*material, "kd", &v) && v->type == JsonValue::Type::Number)
                obj.material.kd = static_cast<float>(v->num);
            if (json_get(*material, "ks", &v) && v->type == JsonValue::Type::Number)
                obj.material.ks = static_cast<float>(v->num);
            if (json_get(*material, "shininess", &v) && v->type == JsonValue::Type::Number)
                obj.material.shininess = static_cast<float>(v->num);
            if (json_get(*material, "kr", &v) && v->type == JsonValue::Type::Number)
                obj.material.kr = static_cast<float>(v->num);
            if (json_get(*material, "ior", &v) && v->type == JsonValue::Type::Number)
                obj.material.ior = static_cast<float>(v->num);

            // NEW: texture path overrides in scene JSON
            if (json_get(*material, "diffuse_texture", &v) && v->type == JsonValue::Type::String)
                obj.diffuse_tex_path = v->str;
            if (json_get(*material, "normal_texture", &v) && v->type == JsonValue::Type::String)
                obj.normal_tex_path = v->str;
        }

        // NEW: per-object "use_mtl" flag (default true)
        if (json_get(item, "use_mtl", &v) && v->type == JsonValue::Type::Bool)
            obj.use_mtl = v->b;

        // NEW: volume-only flag
        if (obj.type == "volume") obj.is_volume = true;

        // NEW: per-object participating medium
        const JsonValue* medObj = nullptr;
        if (json_get(item, "medium", &medObj) && medObj->type == JsonValue::Type::Object) {
            parse_medium(*medObj, obj.medium);
        }

        // Handle volume_region entries separately
        if (obj.type == "volume_region") {
            VolumeRegion vol;
            Vec3 min_b = make_vec3(-1, -1, -1);
            Vec3 max_b = make_vec3(1, 1, 1);
            if (json_get(item, "bounds_min", &v)) json_as_vec3(*v, min_b);
            if (json_get(item, "bounds_max", &v)) json_as_vec3(*v, max_b);
            vol.min_bounds = min_b;
            vol.max_bounds = max_b;
            if (json_get(item, "medium", &medObj) && medObj->type == JsonValue::Type::Object) {
                parse_medium(*medObj, vol.medium);
            }
            scene.volumes.push_back(vol);
        } else if (!obj.path.empty() || obj.is_volume) {
            scene.objects.push_back(obj);
        }
    }

    if (scene.objects.empty() && scene.volumes.empty()) {
        if (err) *err = "Scene contains no valid objects or volumes";
        return false;
    }
    return true;
}

inline bool LoadSceneFromFile(const std::string& path, Scene& scene, std::string* err) {
    std::ifstream f(path);
    if (!f) { if (err) *err = "Failed to open scene file: " + path; return false; }
    std::stringstream buffer;
    buffer << f.rdbuf();
    JsonValue root;
    if (!parse_json(buffer.str(), root, err)) return false;
    return parse_scene(root, scene, err);
}

inline std::string dirname(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

inline bool is_abs_path(const std::string& path) {
    if (path.empty()) return false;
    if (path[0] == '/' || path[0] == '\\') return true;
    if (path.size() >= 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':') return true;
    return false;
}

inline std::string join_path(const std::string& base, const std::string& rel) {
    if (base.empty() || base == ".") return rel;
    if (!base.empty() && (base.back() == '/' || base.back() == '\\')) return base + rel;
    return base + "/" + rel;
}

} // namespace SceneIO