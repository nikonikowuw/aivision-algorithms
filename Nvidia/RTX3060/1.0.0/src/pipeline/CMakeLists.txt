# pipeline/CMakeLists.txt
add_library(algo_pipeline STATIC
    algorithm_context.cpp
    batch_collector.cpp
    tracker.cpp
    face_index.cpp
    thread_pool.cpp
)

target_include_directories(algo_pipeline PUBLIC
    ${ALGO_COMMON_INCLUDES}
    ${PROJECT_SOURCE_DIR}/../../../../engine/include
)

target_link_libraries(algo_pipeline PUBLIC
    algo_common
    algo_runtime
    algo_preprocess
    algo_postprocess
    algo_models
)

# Link FAISS if available
if(HAS_FAISS AND FAISS_LIBRARY)
    target_link_libraries(algo_pipeline PUBLIC ${FAISS_LIBRARY})
    if(BLASFEO_LIBRARY)
        target_link_libraries(algo_pipeline PUBLIC ${BLASFEO_LIBRARY})
    endif()
    if(LAPACK_LIBRARY)
        target_link_libraries(algo_pipeline PUBLIC ${LAPACK_LIBRARY})
    endif()
    target_compile_definitions(algo_pipeline PUBLIC HAS_FAISS=1)
endif()
