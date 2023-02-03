set(hiredis_CXXFLAGS "-Wl,-z,now -s")
set(hiredis_CFLAGS "-Wl,-z,now -s")
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/hiredis/repository/archive/release/v1.0.2.tar.gz")
    set(SHA256 "4d5b92dd338f7c2157844892bc8098a7f966e67b6fa2e75b0b398d3fe5c90eea")
else()
    set(REQ_URL "https://github.com/redis/hiredis/archive/refs/tags/v1.0.2.tar.gz")
    set(SHA256 "e0ab696e2f07deb4252dda45b703d09854e53b9703c7d52182ce5a22616c3819")
endif()

mindspore_add_pkg(hiredis
        VER 1.0.2
        LIBS hiredis hiredis_ssl
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DENABLE_SSL:BOOL=ON
        -DDISABLE_TESTS:BOOL=ON -DOPENSSL_ROOT_DIR:PATH=${openssl_ROOT})

include_directories(${hiredis_INC})
add_library(mindspore_federated::hiredis ALIAS hiredis::hiredis)
add_library(mindspore_federated::hiredis_ssl ALIAS hiredis::hiredis_ssl)
