# VTSCloud TLS trust bundle

`vtscloud-ca.pem` contains public certificates for GlobalSign GCC R46 AlphaSSL
CA 2025 and GlobalSign Root R46. They were exported from the certificate chain
validated by Windows for `rec01ihanoi.vtscloud.vn` on 5 October 2026. The leaf
certificate matched the user-supplied capture. No private key or session data
is included.

The iHanoi adapter loads this bundle as additional trusted CAs using a normal
verified SSL context. Certificate-chain and hostname verification remain
enabled. `--ihanoi-ca-file PATH` selects another additional CA bundle if needed.
