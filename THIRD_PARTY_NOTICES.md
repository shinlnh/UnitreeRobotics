# Third-party notices

The project communicates with NVIDIA Isaac GR00T using the public ZeroMQ
protocol implemented by `gr00t.policy.server_client` at commit
`b9955401d50c92a29258732e3ad6ccd579f1bdc0`. The dependency-light serializer
and client in `unitree_rl_groot/groot/client.py` are an independently reduced
implementation of that protocol. NVIDIA Isaac GR00T is licensed under the
Apache License 2.0; copyright NVIDIA CORPORATION & AFFILIATES.

Isaac Lab-derived configuration patterns are used under Isaac Lab's BSD
3-Clause license. Isaac Sim and downloaded NVIDIA assets retain their own
licenses and EULA.

The optional `datasets/sonic/g1_fetch_clean` training corpus is NVIDIA's
`GR00T-N1.7-AppleToPlate` dataset, pinned by commit in `versions.env` and
licensed under Creative Commons Attribution 4.0 (CC-BY-4.0). Copyright NVIDIA
Corporation.

The optional `checkpoints/nvidia-gr00t-n1.7-applepnp-v1` deployment bundle is
NVIDIA's `GR00T-N1.7-ApplePnP-V1`, pinned in `versions.env` and governed by the
NVIDIA Open Model License Agreement included with the downloaded artifact.

LeApp is consumed from NVIDIA's repository under Apache-2.0. The GEAR whole-body
controller and RoboCasa integration come from NVIDIA GR00T-WholeBodyControl.
The simulation uses the G1-enabled robosuite fork and exact commit named in
`versions.env`; robosuite is distributed under its upstream license.
