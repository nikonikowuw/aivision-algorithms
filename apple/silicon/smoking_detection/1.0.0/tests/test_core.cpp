// Portable smoking detection contract tests; no Apple frameworks required.

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "common/config.h"
#include "common/event_json.h"
#include "common/types.h"
#include "event/event_state.h"
#include "picojson.h"
#include "postprocess/damoyolo_decoder.h"
#include "tracking/tracker.h"

namespace {

int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void TestConfig() {
    smoking::AlgoConfig config;
    std::string error;
    Expect(smoking::ParseConfig("{}", config, error) ==
               smoking::ErrorCode::kSuccess,
           "default config must parse");
    Expect(config.model_dir == "weights", "model_dir default must match schema");
    Expect(smoking::ParseConfig("{\"max_persons\":2.5}", config, error) !=
               smoking::ErrorCode::kSuccess,
           "integer config fields must reject fractions");
    Expect(smoking::ParseConfig("{\"unknown\":1}", config, error) !=
               smoking::ErrorCode::kSuccess,
           "unknown config fields must be rejected");

    smoking::AlgoConfig regions;
    Expect(smoking::ParseConfig(
               "{\"detection_regions\":[[[0,0],[1,0],[1,1]]]}",
               regions, error) == smoking::ErrorCode::kSuccess,
           "valid normalized polygon must parse");
    Expect(smoking::ParseConfig(
               "{\"detection_regions\":[[[0,0],[2,0],[1,1]]]}",
               regions, error) != smoking::ErrorCode::kSuccess,
           "out-of-range polygon coordinates must fail");
}

void TestGeometryAndJson() {
    smoking::NormalizedRect normalized;
    Expect(smoking::NormalizeRect({-10.0f, 20.0f, 50.0f, 100.0f},
                                  100, 100, normalized),
           "clipped rectangle must normalize");
    Expect(normalized.x == 0.0f && normalized.y == 0.2f &&
               normalized.width == 0.4f && normalized.height == 0.8f,
           "normalization must use original-frame dimensions");

    smoking::SmokingEvent event;
    event.detect_confidence = 1.2f;
    event.person_bbox = {-10.0f, 20.0f, 50.0f, 100.0f};
    event.cigarette_bbox = {95.0f, 95.0f, 20.0f, 20.0f};
    event.track_id = 7;
    event.event_id = "smoking:1:1";
    std::string json;
    std::string error;
    Expect(smoking::SerializeEvents({event}, 100, 100, json, error) ==
               smoking::ErrorCode::kSuccess,
           "valid event must serialize");
    picojson::value parsed;
    Expect(picojson::parse(parsed, json).empty() &&
               parsed.is<picojson::array>() &&
               parsed.get<picojson::array>().size() == 1,
           "event JSON must be a top-level array");

    event.person_bbox.x = std::numeric_limits<float>::quiet_NaN();
    Expect(smoking::SerializeEvents({event}, 100, 100, json, error) ==
               smoking::ErrorCode::kSerializationFailed,
           "non-finite geometry must fail serialization");
}

void TestDecoder() {
    smoking::ImageTransform transform;
    transform.source_region_in_frame = {100.0f, 50.0f, 200.0f, 100.0f};
    transform.scale = 2.0f;
    transform.pad_x = 0.0f;
    transform.pad_y = 100.0f;
    transform.frame_width = 400;
    transform.frame_height = 200;
    const float raw[] = {0.0f, 100.0f, 400.0f, 300.0f, 0.9f, 0.0f};
    const auto detections = smoking::DamoOutputParser::Parse(
        raw, 1, 640, 640, transform);
    Expect(detections.size() == 1, "valid DAMO row must decode");
    if (!detections.empty()) {
        Expect(std::abs(detections[0].bbox.x - 100.0f) < 0.001f &&
                   std::abs(detections[0].bbox.y - 50.0f) < 0.001f,
               "DAMO box must map through the shared transform");
    }
}

void TestTrackerAndEvents() {
    smoking::PersonTracker tracker;
    const auto tracks = tracker.Update({{{10, 10, 100, 200}, 0.9f, 0}}, 0);
    Expect(tracks.size() == 1 && tracks[0].track_id == 0,
           "new track must be returned in its creation frame");

    smoking::EventStateMachine::Config config;
    smoking::EventStateMachine state(config);
    const smoking::RectF person{10, 10, 100, 200};
    const smoking::RectF cigarette{60, 30, 5, 5};
    Expect(state.ProcessTrack(1, true, person, cigarette, 0.8f, 0).track_id < 0,
           "first hit must not confirm");
    Expect(state.ProcessTrack(1, false, person, {}, 0.0f, 100).track_id < 0,
           "miss inside candidate window must not confirm");
    Expect(state.ProcessTrack(1, true, person, cigarette, 0.8f, 200).track_id < 0,
           "second hit must not confirm");
    const auto confirmed =
        state.ProcessTrack(1, true, person, cigarette, 0.8f, 300);
    Expect(confirmed.track_id == 1, "third hit in five evaluations must confirm");
    Expect(state.ProcessTrack(1, true, person, cigarette, 0.8f, 400).track_id < 0,
           "latched event must not repeat");

    state.RetireMissingTracks({}, 500);
    const auto inherited =
        state.ProcessTrack(2, true, person, cigarette, 0.8f, 600);
    Expect(inherited.track_id < 0,
           "replacement track must inherit a confirmed lock without double count");
    const auto* inherited_state = state.GetTrackState(2);
    Expect(inherited_state &&
               inherited_state->state == smoking::EventState::kConfirmedLatched,
           "replacement track must remain latched");
}

}  // namespace

int main() {
    TestConfig();
    TestGeometryAndJson();
    TestDecoder();
    TestTrackerAndEvents();
    if (failures != 0) return 1;
    std::cout << "portable smoking detection tests passed\n";
    return 0;
}
