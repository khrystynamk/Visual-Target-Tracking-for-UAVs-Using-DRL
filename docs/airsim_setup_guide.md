# AirSim Setup Guide (macOS)

Building AirSim with Unreal Engine 4.27 on macOS (Sequoia).

## Prerequisites

- macOS with Xcode Command Line Tools (`xcode-select --install`)
- CMake (`brew install cmake`)
- Git

## 1. Install Unreal Engine 4.27

Unreal Engine and any version of it can be installed in the Epic Games launcher, which subsequently can be installed here: [Epic Games](https://store.epicgames.com/en-US/download).

After building, the engine binary will be at:
```
/path/to/UE_4.27/Engine/Binaries/Mac/UE4Editor.app
```

## 2. Clone and Build AirSim

More detailed guide can be found here: [Microsoft AirSim](https://microsoft.github.io/AirSim/build_macos/).

```bash
git clone https://github.com/microsoft/AirSim.git
cd AirSim
./setup.sh
./build.sh
```

### Fix: CMake minimum version error

In `cmake/MavLinkCom/MavLinkTest/CMakeLists.txt`, change the first line to:

```cmake
cmake_minimum_required(VERSION 3.5)
```

This fixes a build error on newer CMake versions.

## 3. Fix Build Targets for Newer macOS/Clang

Newer macOS compilers produce warnings-as-errors that didn't exist when AirSim was written. Fix the Blocks project build targets:

### `Unreal/Environments/Blocks/Source/BlocksEditor.Target.cs`

```csharp
using UnrealBuildTool;
using System.Collections.Generic;

public class BlocksEditorTarget : TargetRules
{
    public BlocksEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        ExtraModuleNames.AddRange(new string[] { "Blocks" });
        DefaultBuildSettings = BuildSettingsVersion.V2;
        bOverrideBuildEnvironment = true;
        AdditionalCompilerArguments = "-Wno-error -Wno-bitwise-instead-of-logical -Wno-deprecated-builtins";
    }
}
```

### `Unreal/Environments/Blocks/Source/Blocks.Target.cs`

```csharp
using UnrealBuildTool;
using System.Collections.Generic;

public class BlocksTarget : TargetRules
{
    public BlocksTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        ExtraModuleNames.AddRange(new string[] { "Blocks" });
        if (Target.Platform == UnrealTargetPlatform.Linux)
            bUsePCHFiles = false;
        bOverrideBuildEnvironment = true;
        AdditionalCompilerArguments = "-Wno-error -Wno-bitwise-instead-of-logical -Wno-deprecated-builtins";
    }
}
```

The key additions are `bOverrideBuildEnvironment = true` and the `AdditionalCompilerArguments` that suppress Clang warnings treated as errors on newer macOS.

## 4. Change the Environment Scene (Optional)

To use a custom map instead of the default Blocks scene, edit:

**`Unreal/Environments/Blocks/Config/DefaultEngine.ini`:**

```ini
[/Script/EngineSettings.GameMapsSettings]
EditorStartupMap=/Game/Stylized_PBR_Nature/Maps/Stylized_Nature_ExampleScene.Stylized_Nature_ExampleScene
LocalMapOptions=
TransitionMap=
bUseSplitscreen=True
TwoPlayerSplitscreenLayout=Horizontal
ThreePlayerSplitscreenLayout=FavorTop
GameInstanceClass=/Script/Engine.GameInstance
GameDefaultMap=/Game/Stylized_PBR_Nature/Maps/Stylized_Nature_ExampleScene.Stylized_Nature_ExampleScene
ServerDefaultMap=/Engine/Maps/Entry
GlobalDefaultGameMode=/Script/AirSim.AirSimGameMode
GlobalDefaultServerGameMode=None
```

Replace the map paths (`EditorStartupMap` and `GameDefaultMap`) with the path to your desired map. This can be also done in the editor, but I found this way to be more convenient. `GlobalDefaultGameMode` must point to `/Script/AirSim.AirSimGameMode` for AirSim to work.

## 5. Launch the Simulator

### Without opening the Unreal Editor

```bash
/path/to/UE_4.27/Engine/Binaries/Mac/UE4Editor.app/Contents/MacOS/UE4Editor \
  /path/to/AirSim/Unreal/Environments/Blocks/Blocks.uproject \
  -game -log
```

## 6. AirSim Settings

AirSim reads its configuration from `~/Documents/AirSim/settings.json`. This file is created automatically on first launch with default settings.

For this project, copy the project's settings:
```bash
cp configs/airsim/settings.json ~/Documents/AirSim/settings.json
```

Changes to settings require restarting UE4.

## Troubleshooting

| Issue | Solution |
|---|---|
| Build fails with `-Wbitwise-instead-of-logical` | Apply the build target fixes from Step 3 |
| `cmake_minimum_required` error | Apply the CMake fix from Step 2 |
| Drones spawn in wrong location | Check that `~/Documents/AirSim/settings.json` matches `configs/airsim/settings.json` and restart UE4. Also check the `PlayerStart` actor in the UE editor and move it to the desired location |
