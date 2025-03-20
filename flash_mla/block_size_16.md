    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         2.62
    SM Frequency            cycle/nsecond         1.65
    Elapsed Cycles                  cycle       589980
    Memory Throughput                   %        33.90
    DRAM Throughput                     %        21.27
    Duration                      usecond       357.25
    L1/TEX Cache Throughput             %        35.09
    L2 Cache Throughput                 %        23.01
    SM Active Cycles                cycle    569475.24
    Compute (SM) Throughput             %        77.56
    ----------------------- ------------- ------------

    OPT   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the   
          compute pipelines are spending their time doing. Also, consider whether any computation is redundant and      
          could be reduced or moved to look-up tables.                                                                  

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
    Shared Memory Configuration Size           Kbyte          135.17
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block          133.12
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
    Achieved Occupancy                        %        12.48
    Achieved Active Warps Per SM           warp         7.98
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 87.5%                                                                                     
          The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 16. This kernel's theoretical occupancy (12.5%) is limited by the number of required      
          registers. This kernel's theoretical occupancy (12.5%) is limited by the required amount of shared memory.    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       198987
    Total DRAM Elapsed Cycles        cycle     44907008
    Average L1 Active Cycles         cycle    569475.24
    Total L1 Elapsed Cycles          cycle     45967996
    Average L2 Active Cycles         cycle    518521.18
    Total L2 Elapsed Cycles          cycle     58500480
    Average SM Active Cycles         cycle    569475.24
    Total SM Elapsed Cycles          cycle     45967996
    Average SMSP Active Cycles       cycle    569719.04
    Total SMSP Elapsed Cycles        cycle    183871984
    -------------------------- ----------- ------------
