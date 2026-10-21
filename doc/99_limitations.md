# Current Limitations

- So far, our framework is limited to:
  - networks using `Ns3::PointToPointConnection` links,
  - deterministic unicast traffic,
  - TCP and UDP traffic only
  - Up to 65533 flows per source node,
  - TCP without SACK.
- Training and simulation are mainly CPU-bound tasks, so strong CPUs with many cores are recommended, 
  especially if you are starting many experiments at once.
- It is recommended to use at least 4 CPU cores per run.
- GPU training is implemented but slow. We are working on improving this.

Further limitations can be derived from our extensive list of Github issues.