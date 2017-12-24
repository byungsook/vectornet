cmake_minimum_required(VERSION 2.8)

project(gco)

SET(SOURCES
	LinkedBlockList.cpp
    graph.cpp
    maxflow.cpp
    QPBO.cpp
    QPBO_extra.cpp
    QPBO_maxflow.cpp
    QPBO_postprocessing.cpp
    GCoptimization.cpp
    main.cpp
)

SET(HEADERS
	LinkedBlockList.h
    graph.h
    block.h
    QPBO.h
    GCoptimization.h
)

include_directories("./")
add_executable(gco ${SOURCES} ${HEADERS})
