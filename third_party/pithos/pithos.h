#ifndef __PITHOS_H
#define __PITHOS_H

#include <graal_isolate.h>


#if defined(__cplusplus)
extern "C" {
#endif

int vdb_init(graal_isolatethread_t*);

int vdb_load_index(graal_isolatethread_t*, char*, char*);

int vdb_load_index_with_weights(graal_isolatethread_t*, char*, char*, float*, int);

int vdb_get_info(graal_isolatethread_t*, char*, int*, long long*, char*, long long*, int*);

int vdb_batch_search(graal_isolatethread_t*, char*, float*, int, int, long long*, int*);

long long int vdb_query_planetary_grid(graal_isolatethread_t*, char*, float*, int*, int*, int, char*);

int vdb_compile_index_file(graal_isolatethread_t*, char*, char, long long int, int, int*, int, long long*, float*, int, int);

long long int vdb_size(graal_isolatethread_t*, char*);

int vdb_drop_index(graal_isolatethread_t*, char*);

int vdb_set_chunk_size(graal_isolatethread_t*, char*, long long int);

int vdb_set_energy_budget(graal_isolatethread_t*, char*, double);

int vdb_get_tier_address(graal_isolatethread_t*, char*, int, long long*, long long*);

int vdb_transform_and_quantize(graal_isolatethread_t*, char*, float*, long long*);

int vdb_close(graal_isolatethread_t*);

int vdb_create_delta_buffer(graal_isolatethread_t*, char*, int);

int vdb_insert(graal_isolatethread_t*, char*, long long int, float*);

int vdb_delete_from_delta(graal_isolatethread_t*, char*, long long int);

long long int vdb_delta_size(graal_isolatethread_t*, char*);

int vdb_needs_flush(graal_isolatethread_t*, char*);

int vdb_search_merged(graal_isolatethread_t*, char*, float*, int, long long*, int*);

int vdb_backup_delta(graal_isolatethread_t*, char*, char*);

int vdb_restore_delta(graal_isolatethread_t*, char*, char*, int);

int vdb_cuda_init(graal_isolatethread_t*, int);

int vdb_cuda_shutdown(graal_isolatethread_t*);

int vdb_cuda_is_available(graal_isolatethread_t*);

int vdb_cuda_batch_search(graal_isolatethread_t*, char*, float*, int, int, long long*, int*);

long long int vdb_cuda_query_planetary_grid(graal_isolatethread_t*, char*, float*, int*, int*, int, char*);

#if defined(__cplusplus)
}
#endif
#endif
