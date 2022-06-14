if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/hiredis/repository/archive/release/v1.0.2.tar.gz")
    set(MD5 "20acecf5cb87868723f97a9bbb5cf045")
else()
    set(REQ_URL "https://github.com/redis/hiredis/archive/refs/tags/v1.0.2.tar.gz")
    set(MD5 "58e8313188f66ed1be1c220d14a7752e")
endif()

mindspore_add_pkg(hiredis
        VER 1.0.2
        LIBS hiredis hiredis_ssl
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DENABLE_SSL:BOOL=ON
        -DDISABLE_TESTS:BOOL=ON)

include_directories(${hiredis_INC})
add_library(mindspore_federated::hiredis ALIAS hiredis::hiredis)
add_library(mindspore_federated::hiredis_ssl ALIAS hiredis::hiredis_ssl)
