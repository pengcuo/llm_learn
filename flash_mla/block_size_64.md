
   Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         2.62
    SM Frequency            cycle/nsecond         1.64
    Elapsed Cycles                  cycle       534862
    Memory Throughput                   %        23.38
    DRAM Throughput                     %        23.38
    Duration                      usecond       324.80
    L1/TEX Cache Throughput             %        19.35
    L2 Cache Throughput                 %        23.98
    SM Active Cycles                cycle    518383.56
    Compute (SM) Throughput             %        85.58
    ----------------------- ------------- ------------

    INF   The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To   
          further improve performance, work will likely need to be shifted from the most utilized to another unit.      
          Start by analyzing workloads in the Compute Workload Analysis section.                                        

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     78
    Registers Per Thread             register/thread             255
    Shared Memory Configuration Size           Kbyte          233.47
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block          230.40
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              78
    Threads                                   thread           19968
    Uses Green Context                                             0
    Waves Per SM                                                   1
    -------------------------------- --------------- ---------------

    OPT   If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have more than the 
          achieved 1 blocks per multiprocessor. This way, blocks that aren't waiting for __syncthreads() can keep the   
          hardware busy.                                                                                                

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit SM                        block           32
    Block Limit Registers                 block            1
    Block Limit Shared Mem                block            1
    Block Limit Warps                     block            8
    Theoretical Active Warps per SM        warp            8
    Theoretical Occupancy                     %        12.50
    Achieved Occupancy                        %        12.44
    Achieved Active Warps Per SM           warp         7.96
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 87.5%                                                                                     
          The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 16. This kernel's theoretical occupancy (12.5%) is limited by the number of required      
          registers. This kernel's theoretical occupancy (12.5%) is limited by the required amount of shared memory.    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle    198841.50
    Total DRAM Elapsed Cycles        cycle     40826880
    Average L1 Active Cycles         cycle    518383.56
    Total L1 Elapsed Cycles          cycle     41656416
    Average L2 Active Cycles         cycle    245886.29
    Total L2 Elapsed Cycles          cycle     52900128
    Average SM Active Cycles         cycle    518383.56
    Total SM Elapsed Cycles          cycle     41656416
    Average SMSP Active Cycles       cycle    516604.19
    Total SMSP Elapsed Cycles        cycle    166625664
    -------------------------- ----------- ------------
