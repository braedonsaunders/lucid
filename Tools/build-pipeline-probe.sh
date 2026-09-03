#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
mkdir -p .build/bin
swiftc -O -parse-as-library \
  -module-cache-path .build/probe-module-cache -Xcc -fmodules-cache-path=.build/xcode-module-cache \
  Lucid/Metal/MetalTileCompositor.swift Lucid/Metal/SuperResolutionSession.swift \
  Lucid/Metal/TiledVideoToolboxUpscaler.swift Lucid/Pipeline/EnhancementPipeline.swift \
  Lucid/Capture/CapturedFrame.swift Lucid/Overlay/FramePresenter.swift \
  Tools/PipelineProbe.swift -o .build/bin/pipeline-probe
