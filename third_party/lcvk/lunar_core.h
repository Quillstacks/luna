#ifndef __LUNAR_CORE_H
#define __LUNAR_CORE_H

#include <graal_isolate.h>


#if defined(__cplusplus)
extern "C" {
#endif

int vdb_init(graal_isolatethread_t*);

int vdb_load_index(graal_isolatethread_t*, char*, char*);

int vdb_batch_search(graal_isolatethread_t*, char*, long long*, int, int, long long*, int*);

long long int vdb_query_planetary_grid(graal_isolatethread_t*, char*, long long*, int*, int*, int, char*);

int vdb_compile_index_file(graal_isolatethread_t*, char*, char, long long int, long long*, long long*, int);

long long int vdb_size(graal_isolatethread_t*, char*);

int vdb_drop_index(graal_isolatethread_t*, char*);

int vdb_set_chunk_size(graal_isolatethread_t*, char*, long long int);

int vdb_close(graal_isolatethread_t*);

#if defined(__cplusplus)
}
#endif
#endif
