# Paging

Paging is a memory-management scheme that permits the physical address space of a process to be noncontiguous. Physical memory is broken into fixed-sized blocks called frames and logical memory into blocks of the same size called pages. A page table maps each page number to a frame number. Paging avoids external fragmentation but may suffer internal fragmentation in the last page. The translation look-aside buffer (TLB) is a small fast associative memory that caches recent page-table entries to speed up address translation.
