set(protobuf_USE_STATIC_LIBS ON)
if(BUILD_LITE)
    if(MSVC)
        set(protobuf_CXXFLAGS "${CMAKE_CXX_FLAGS}")
        set(protobuf_CFLAGS "${CMAKE_C_FLAGS}")
        set(protobuf_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        set(_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
        set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
    else()
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_CXXFLAGS "${protobuf_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
        set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
    endif()
else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
            -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    else()
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_CXXFLAGS "${protobuf_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
    endif()
    set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v3.13.0.tar.gz")
    set(MD5 "53ab10736257b3c61749de9800b8ce97")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz")
    set(MD5 "1a6274bc4a65b55a6fa70e264d796490")
endif()

set(PROTOBUF_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/protobuf)

mindspore_add_pkg(protobuf
        VER 3.13.0
        LIBS protobuf
        EXE protoc
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_PATH cmake/
        CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
        PATCHES ${PROTOBUF_PATCH_ROOT}/CVE-2021-22570.patch)

include_directories(${protobuf_INC})
add_library(mindspore_federated::protobuf ALIAS protobuf::protobuf)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()

if(EXISTS ${protobuf_ROOT}/lib64)
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${protobuf_ROOT}/lib64/cmake/protobuf")
else()
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${protobuf_ROOT}/lib/cmake/protobuf")
endif()
message("Using Protobuf_DIR : " ${_FINDPACKAGE_PROTOBUF_CONFIG_DIR})

function(ms_protobuf_generate c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ms_proto_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})

    foreach(proto_file_with_path ${ARGN})
        message(proto_file_with_path: ${proto_file_with_path})
        get_filename_component(proto_file_absolute "${proto_file_with_path}" ABSOLUTE)
        message(proto_file_absolute: ${proto_file_absolute})
        get_filename_component(file_dir ${proto_file_absolute} DIRECTORY)
        get_filename_component(proto_I_DIR "${file_dir}/../../" ABSOLUTE)
        get_filename_component(proto_file ${proto_file_absolute} NAME)
        get_filename_component(proto_file_prefix ${proto_file_absolute} NAME_WE)

        set(protoc_output_prefix ${CMAKE_BINARY_DIR}/common/protos)
        set(hw_proto_srcs "${protoc_output_prefix}/${proto_file_prefix}.pb.cc")
        set(hw_proto_hdrs "${protoc_output_prefix}/${proto_file_prefix}.pb.h")
        add_custom_command(
                OUTPUT ${hw_proto_srcs} ${hw_proto_hdrs}
                WORKING_DIRECTORY ${proto_I_DIR}
                COMMAND $<TARGET_FILE:protobuf::protoc>
                ARGS --cpp_out "${CMAKE_BINARY_DIR}"
                -I "${proto_I_DIR}"
                "${proto_file_absolute}"
                COMMAND $<TARGET_FILE:protobuf::protoc>
                ARGS --python_out "${CMAKE_BINARY_DIR}"
                -I "${proto_I_DIR}"
                "${proto_file_absolute}"
                DEPENDS "${proto_file_absolute}")

        list(APPEND ${c_var} ${hw_proto_srcs})
        list(APPEND ${h_var} ${hw_proto_hdrs})
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()
