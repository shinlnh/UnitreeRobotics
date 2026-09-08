# Hugging Face artifact mirrors

Large artifacts that do not belong in GitHub are mirrored to these canonical
Hugging Face repositories:

- Models: <https://huggingface.co/shin0412/UnitreeRobotic>
- Benchmarks and other large data: <https://huggingface.co/datasets/shin0412/UnitreeRobotic>

The Hugging Face branch graph follows the GitHub feature graph: A0 and A1
branch from `main`, A2 branches from A1, B branches from A2, and B-retry and
Ours branch from B. Commit titles use the same Conventional Commit style as
the corresponding GitHub artifact commit. Full GitHub source SHAs are stored
in the Hugging Face commit messages.

| Feature | GitHub source | Model commit | Dataset commit | Dataset payload added |
| --- | --- | --- | --- | ---: |
| A0 | `7ecf86259a8d800a1506fa9dcb73a5832b44f340` | `e87de28f85296855b8adcbeb8ba419f6a47375f9` | `c5825dff3f92f4ca276a5375968f1b83310eab6d` | 28,388,039,947 B |
| A1 | `68b1a1f39570731ded4b9d56c1d017dbe0f29478` | `11b5e08e71431035c75d3eeeee2bc6731725abca` | `b763ba56a08052406cb4ceb2856fb53c7fac4080` | 31,317,064,265 B |
| A2 | `c9930fd0aa9ffec8d4fde817981680955ff9281f` | `6cfca341d8c9f2622aff7f59dfe0738112a0ed33` | `d7d0f516862720d1849baf2fa83d9d282be7f9d0` | 30,902,981,382 B |
| B | `06ae5d385ad55415332e4f1688b7cd59f868ac96` | `34235eed4ad8cf2724a691a22dd28b93d0eb64a8` | `be9e12b06599e5916c582fe8e86ab62c2699e1b6` | 7,258,466,080 B |
| Ours R0 | `b69ec837d088709f59015f46ee7742475b429022` | `ab8f52158020041eb36d4cf91924ea2c5c1232ce` | `dd29f40bfe20c13e107ed4fe3e4557d22e3431d3` | 86,859,304 B |

The Ours entry is an intermediate, train-only R0 mirror. Development and final
rollout artifacts are added only after their respective protocol gates.

## Branches

| Feature | Hugging Face branch |
| --- | --- |
| A0 | `feat(A0)/implement-GR00T-N1.7-LIBERO-original` |
| A1 | `feat(A1)/implement-GR00T-RC` |
| A2 | `feat(A2)/implement-GR00T-RC-fixed-hierarchy` |
| B | `feat(B)/implement-GR00T-RC-SparkVLA-style-execution` |
| Ours | `feat(Ours)/implement-learned-recovery` |

## Layout and restoration

Model weights retain their repository-relative checkpoint paths. Dataset
decision logs are stored unchanged at
`artifacts/<feature>/full-benchmark/<horizon>/decisions.jsonl`. Frame trees are
packed losslessly into one uncompressed tar per condition at
`frames-archives/<condition>.tar` to avoid hundreds of thousands of individual
Hub files.

Restore an archive from the directory that should receive the condition folder:

```bash
tar -xf frames-archives/Ideal.tar
```

A1's converted LeRobot tree is stored as
`outputs/robocerebra/gr00t-lerobot-v2.tar`. B's selector feature cache is stored
as `outputs/robocerebra/sparkvla-selector-v1/features.tar`. Both archives are
uncompressed and preserve their original directory trees.

## Integrity checksums

Model LFS objects:

| Artifact | SHA-256 |
| --- | --- |
| A0 model shard 1 | `39b7beaadd9a06c87502f2b741a36f939b84941f37fef757c6177e54e34a5eff` |
| A0 model shard 2 | `8c4fa56f1b4f25a1a842c811e8a12d2bf6d77d73942f9a04f5b876a11bd6d703` |
| A1/A2/B base model shard 1 | `d273ca475107b081cb91ecb4f555b6ac46692ba0d8c337935c00a03dc889b973` |
| A1/A2/B base model shard 2 | `244a7ea7e7b2510a31cad92a805f59651301277a9e36508f863012bde49ed4aa` |
| B selector | `88a05811257384bbba867169b72103d3bc1e118f5813a9d6c4f4713b274e1f16` |
| Ours R0 MLP H16 v00 | `f51a4f2bab79d025be1568af828fd3ad232001539bb82abc1a6f318d2431fe03` |
| Ours R0 MLP H16 v01 | `6f2655b2ea43beb63ce43b72f2607383e66456cb1a1277f342ecf0cc6e3e1ad1` |
| Ours R0 GRU H16 v02 | `2761206c91bbab52896e29ab469613f99de5af0181d94e3e2b61c10ab1dbca12` |
| Ours R0 Transformer H8 v03 | `db0b846dbd7be90cc75c201b1475cb1e4d19f1e9ac9f97a08f907d224cc6a4a8` |

Decision logs and auxiliary archives:

| Artifact | SHA-256 |
| --- | --- |
| A0 H16 decisions | `f8ddcc82525ca0c3ecaa8da5e868e254920ec606807e2dd4f4db159b2081fecd` |
| A0 H8 decisions | `ba761aabb628ec1b3c009972cc2eedb0bd38a0667a9ef03ba1375ad61d12a29d` |
| A1 H16 decisions | `e649b283adb667236a763755203631ee63b79f2754fdeb6eae49731ddb547fbb` |
| A1 H8 decisions | `b14886a2260220bd3f4a606f07d1272abbf7439b33657caef64a025b60b06835` |
| A1 converted LeRobot archive | `4648dac64df2c4732904dcc9aca8cd45c235818f98c15cf0e0254dd0f82234c2` |
| A2 H16 decisions | `13d2d730cbacf7db39f097ac74075d343514444380a5309f2e3cb14e6a8bac4d` |
| A2 H8 decisions | `7d97d107a0432b64fe57694be5980a221d3bec3a188ba8b8338b2c7d9631cfa1` |
| B H16 decisions | `f8d24466d24973cf0155632ed642b68e47811689de2329271f909dcc20de4d33` |
| B H8 decisions | `1878426a806d34f595571d7162d60fda222e9a7dfd7f928f622180f0dc518121` |
| B selector feature archive | `173325dd3a697e6dfe3e640eb26aeb59366de14c500faf275b57060e67b56e88` |
| Ours R0 counterfactual corpus manifest | `1f8d2040ee2ba9913608b6ad2003a6e4cfa681b62b31a57a4d6e3f9505ab075b` |
| Ours R0 counterfactual branch log | `373f494d8435f8a61e3a05a256276af9a1bb63a92f878f37b1bedf54fa74b399` |

Frame archive SHA-256 values, ordered as `Ideal`, `Memory_Execution`,
`Memory_Exploration`, `Mix`, `Observation_Mismatching`, and
`Random_Disturbance`:

| Feature/horizon | SHA-256 values in condition order |
| --- | --- |
| A0 H16 | `62cbc9aca626854ed1c338768d4d6481c98d111a9f3454c19cd947c88e05fae2`, `97c6e409f53539b53e2a65a78b25957e8043b3179f8ad7a1680a221f6c40ae9d`, `51f849c9decb5719b8ed98dbcfa0e29d359b55bcecd7aaca7dd8666de03fe8c7`, `97460c29fdd9a09227f2007ae04ad9f2a6841ce357f29e3d6a64444d2393513c`, `3391bbc43cd6000b5e73ebc6e0ca60a3322259b68caec6ca4f60d760fb176c8a`, `0765390629ffb40ce932f50c74dddba1d1ebe2ccd1ef1e91feae9e53c420495e` |
| A0 H8 | `6f8fde2e57737a027f4bda7f8ac296789d73c0c16ca6b91f7318bc99ebf55a63`, `0d7bbeb3eba919703092ffe0aae85defcc21a429dcf03838e94002375e4130a2`, `b14bef5e79c4766184b6634a462e802592c17174bb76d40c01916ac614348c36`, `93860fc2e6277fa9af5ef9dbced1a7bb5c1f91d783d65a508691663f474d4052`, `abf1cf8ad3f2a31de7b1d0cab7814e78522a5881b2206956775aec9c6830ae35`, `374ef88e1807647bd41499889d500b7339f57dea4831e247d036a85eb051f4f3` |
| A1 H16 | `b8a9a41565261fc9189f141f3347a6186c2bbe266436b7a7a4666a484675454b`, `c9a970efa4723d556087ca7df34d384c0d1ec603dab0a59c4eec4ff397ddda01`, `630a22a576a89d2d9ec9c6e529f3f2af19c0bff43a3a39def893919e57e4c808`, `38d4ecfc42c02ac460e6aef9d283fe8864c9318d594a365362b9f9816dcad465`, `ac6f06605892c62a65a59adef389066991c94b5c4bd417c4f054662343368a32`, `dccf82659cbd1aafb18f5ed1c6c15561501a6f0cbe659c53ed114931396442c0` |
| A1 H8 | `fd31203b9c10c524b7b929cd3566df521791af0ec88dc5b20a11d2a7e278b08e`, `74f4c7b4234dd4ea24ff32d287c206297941c6f39766468e88f452d3b73456e1`, `957e1ca2d966db61aade62f6fd5d3b862dc6f7d7c0b04e7ef5ba62fa1209e2d4`, `e375c40a4c2e5a866b6e1c20d396900ad2b32ab56d00bbc8bc08fbdbdc2c11a6`, `b00213fe25c430b2ff89f4ba6dc4c5c4940dcffaf474291ccefd4b2095e28e41`, `8d3e26eeb1079b53823302d439f38dcd0f9f9c389512921f48e9faf7a5f77a33` |
| A2 H16 | `3978016cbd640ea5c08e6afb52addee17458caeacd2b0d839dff770ac054f467`, `8ec2c04f5206bb4c8732facec8d49e2bafd4463872667282ee8dcd4b88aec03f`, `481670fb855adca140c5d7f430911feff30ca3a46e1b9ede308eebbef1751e1e`, `868357d22dc66139623ebf54a272d889824b8d2fbd47d23abd8981e1e417cc72`, `17a99e06145c73cbd9276cf53ae7e6f395be4b87f911de4f8feac8040d4e81c2`, `3df8548ed21864c9cf47529399aff38d8a5aa6529f63cb4bd8efb6d4d31485b2` |
| A2 H8 | `85934c7bff6bf06611a801859d0e993fa24ee2fe1f3916a05ae70a845cb89f64`, `562dd702c4bee488067b9b7408431274c3d2f0a67a790d4e0290df8c06165200`, `bb6c669d2b8a8401db13cbe55492ee7d8833e57af374f628ce4c1d19778b7964`, `e3a1fe9f9e077c557ae8000c166cdd5597bbed0fa95da155593b333c238ee7e6`, `3f46f0351a86fe492532424c54ffcaa2483d79bb9076b6ba63023c95bca4a9a1`, `c3e25b8290c2146bce57e544485c9113725322c968b412e0d9850f965667a539` |
| B H16 | `5e5dc0dfcf8f969550454414956be1a6473ba09747412900ef48126c0f5a1a34`, `b95020e73557f99ddfdd64c73cfcca84e1d56372f64bda9c82217da5d2c19b8b`, `a98067835e1a6f29456f264fadf0030a84a2c1ffe89183f6a270aa5470fa7a49`, `18b387417694c53da8d6c758fe65e64d0fad485ef502ff1ca7cd1fb749cb3607`, `0e78aa726c2671407ea16c747324ad48fe2cbaf6b3f3b1ea38d850dc66280f72`, `c8776649be8ffa1e0ffb9a4355b502a6e2adc58fcdcf9c3820211c4dd889c8a8` |
| B H8 | `20f6a409ceeaa97a47ff864ccff51d5df13b7f6a02602afda520f7c8ecc7e17b`, `29d6ebb0ad15af519be535375b2aeca2c4076677e985ef146b29fbb39eb30553`, `26637045b1caeaac264971d1fd2daec70f4cfbcc27c63f2ba1919f7184f14f7e`, `ef9d73723ae8f9ef9c9701fa1e8253afe5e8bf424fad5a2c027bd0bef1d8ac29`, `c58d59a42f7a61b2daedc2fad1c96e106503603247ccf2a84a652558c83dbfac`, `760b5ba1adf0d97db06b9e65cbf46672d7af44173b5a2fb7f355aa7952882d6e` |
