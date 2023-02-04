if(MSVC)
    set(flatbuffers_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_CFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
else()
    set(nlohmann_json_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    set(nlohmann_json_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.6.1.zip")
    set(SHA256 "0857e519a5d86e19bc1f61ce1a330521f224f3c14d7da83fd28e3a1a9804693a")
    set(INCLUDE "./include")
else()
    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip")
    set(SHA256 "69cc88207ce91347ea530b227ff0776db82dcb8de6704e1a3d74f4841bc651cf")
    set(INCLUDE "./")
endif()

mindspore_add_pkg(nlohmann_json
        VER 3.6.1
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        SHA256 ${SHA256})
include_directories(${nlohmann_json_INC})
add_library(mindspore_federated::json ALIAS nlohmann_json)
