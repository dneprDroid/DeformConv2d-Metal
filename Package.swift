// swift-tools-version:5.5

import PackageDescription

let package = Package(
    name: "DeformConv2d-Metal",
    platforms: [
        .iOS(.v15),
        .macOS("11.0")
    ],
    products: [
        .library(
            name: "DeformConv2dMetal",
            type: .dynamic,
            targets: ["DeformConv2dMetal"]
        ),
    ],
    targets: [
        .target(
            name: "DeformConv2dMetal",
            path: "swift/DeformConv2dMetal",
            resources: [
                .copy("Resources/Metal"),
            ]
        )
    ]
)
