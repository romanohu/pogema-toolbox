# MovingAI benchmark data

Source: [Moving AI Lab benchmarks](https://movingai.com/benchmarks/index.html), maintained by Nathan Sturtevant and contributors; [MAPF collection](https://movingai.com/benchmarks/mapf.html).

The source provides the benchmark database under the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/). This concerns database rights; underlying map contents may carry separate rights. The [grid benchmark page](https://movingai.com/benchmarks/grids.html) recognizes BioWare's permission to distribute its maps for research. It also states that explicit redistribution permission has not been obtained for some other map sets. This repository does not represent those maps as Apache licensed or unrestricted. It distributes identities and notices, and prepares original assets locally for research, without committing or packaging raw map/scenario data.

Please cite Nathan Sturtevant, “Benchmarks for Grid-Based Pathfinding,” IEEE Transactions on Computational Intelligence and AI in Games, 4(2), 144–148, 2012. [Paper](https://web.cs.du.edu/~sturtevant/papers/benchmarks.pdf).

Archive URLs, byte counts, archive SHA256s and member SHA256s in `manifest.yaml` identify the verified 33 maps and 25 random plus 25 even scenario files per map. Record these identities and this attribution when sharing derived experimental results. Original map dimensions and borders are preserved; only source coordinates `(x, y)` are translated to `(row=y, column=x)`.
