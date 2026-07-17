add_executable(smoking_unit_tests test_core.cpp)
target_include_directories(smoking_unit_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(smoking_unit_tests PRIVATE
    smoking_common
    smoking_event
    smoking_postprocess
    smoking_tracking)
add_test(NAME smoking_unit_tests COMMAND smoking_unit_tests)
