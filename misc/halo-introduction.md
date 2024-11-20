# Halo 正式开源: 使用可穿戴设备进行开源健康追踪

![](https://files.mdnice.com/user/38198/845ff80c-01e6-40a5-80df-55e308933dd0.jpg)

在飞速发展的可穿戴技术领域，我们正处于一个十字路口。市场上充斥着各式时尚、功能丰富的设备，声称能够彻底改变我们对健康和健身的方式。然而，在这些光鲜的外观和营销宣传背后，隐藏着一个令人担忧的现实：大多数这些设备是封闭系统，其内部运行被专有代码和封闭硬件所掩盖。作为消费者，我们对这些设备如何收集、处理及可能共享我们的健康数据一无所知。

这时，**Halo** 出现了，它是一种旨在让健康追踪更加普惠化的开源替代方案。通过这系列文章，我们将引导你从基础入手，构建并使用完全透明、可定制的可穿戴设备。

需要说明的是，Halo 的目标并不是在抛光度或功能完整性上与消费级可穿戴设备竞争。相反，它提供了一种独特的、动手实践的方式来理解健康追踪设备背后的技术。

我们将使用 `Swift 5` 来构建对应的 iOS 界面，以及 `Python >= 3.10`。由于此项目的代码完全 [开源](https://github.com/cyrilzakka/Halo-iOS)，你可以随时提交合并请求，或者分叉项目以探索全新的方向。

**你将需要：**

- 获取 [COLMI R02](https://www.aliexpress.us/item/3256806445134241.html?gatewayAdapt=glo2usa4itemAdapt) 实际设备，价格在撰写时为 11 到 30 美金左右。
- 一个安装了 Xcode 16 的开发环境，以及可选的 Apple 开发者计划会员资格。
- `Python >= 3.10`，并安装了 `pandas`、`numpy`、`torch` 当然还有 `transformers`。

**致谢**

此项目基于 [Python 仓库](https://tahnok.github.io/colmi_r02_client/) 的代码及我的学习成果构建。

**免责声明**

作为一名医生，我有法律义务提醒你：你即将阅读的内容并不是医学建议。现在，让我们开始让一些可穿戴设备发出蜂鸣声吧！

### 配对戒指

在进入代码之前，让我们先了解蓝牙低能耗（BLE）的关键规格。BLE 基于一个简单的客户端-服务器模型，使用三个核心概念：_中央设备（Centrals）_、_服务（Services）_ 和 _特征（Characteristics）_。以下是它们的具体介绍：

- **中央设备**（例如你的 iPhone）负责启动和管理与**外设**（例如我们的 COLMI R02 戒指）的连接。戒指通过广播自身信息等待手机连接，每次仅支持一台手机连接。
- **服务**是戒指上相关功能的集合，例如心率监测服务或电池状态服务。每个服务都有一个唯一标识符（UUID），客户端通过它来找到对应服务。
- **特征**是每个服务中的具体数据点或控制机制。例如，它们可能是只读（获取传感器数据）、只写（发送命令）或两者兼有。有些特征还能在其值发生变化时自动通知手机，这对于实时健康监测尤为重要。

当手机连接到戒指时，会定位所需的服务，并与特定特征交互以发送命令或接收数据。这种结构化的方法不仅确保了通信效率，还能延长电池使用时间。了解了这些基础知识后，让我们开始构建吧！

### 设置 Xcode 项目

创建一个名为 `Halo` 的新项目，目标平台为 `iOS`。组织标识符建议使用反向域名格式（如 `com.example`）。本项目中，我们使用 `com.FirstNameLastName`。

接下来，为应用启用必要的功能。在 Xcode 中，打开 `Signing & Capabilities` 选项卡，启用以下 **后台模式（Background Modes）**，以确保应用在后台运行时能够保持与戒指的连接并处理数据。

然后，我们将使用 Apple 提供的最新框架 [`AccessorySetupKit`](https://developer.apple.com/documentation/accessorysetupkit/)，用于将蓝牙和 Wi-Fi 配件连接到 iOS 应用。此框架自 iOS 18 推出，替代了传统的广泛蓝牙权限请求方式，专注于为用户明确批准的特定设备提供访问权限。

当用户尝试将 COLMI R02 戒指连接到应用时，`AccessorySetupKit` 会显示一个系统界面，仅列出兼容的附近设备。用户选择设备后，应用即可与戒指通信，而无需请求完整的蓝牙权限。这大大提升了用户隐私，同时简化了设备连接的管理流程。

打开 `Info.plist` 文件（可以在左侧边栏中找到，或通过 `Project Navigator (⌘1) > Your Target > Info` 定位）。添加以下键值条目以支持与 COLMI R02 戒指的配对：

- 添加 `NSAccessorySetupKitSupports`，类型为 `Array`，并将 `Bluetooth` 作为第一个项目。
- 添加 `NSAccessorySetupBluetoothServices`，类型为 `Array`，并将以下 UUID 作为 `String` 项：
  - `6E40FFF0-B5A3-F393-E0A9-E50E24DCCA9E`
  - `0000180A-0000-1000-8000-00805F9B34FB`

至此，初步配置完成！🤗


![](https://files.mdnice.com/user/38198/011bcddb-3c76-4b59-b94a-abf87488cf31.png)

### Ring Session Manager 类

接下来，我们将创建一个 `RingSessionManager` 类，用于管理所有与戒指的通信。此类的主要职责包括：

- 扫描附近的戒指
- 连接到戒指
- 发现服务和特征
- 实现数据读写操作

#### 第一步：创建 RingSessionManager

首先创建一个新的 Swift 文件（⌘N），命名为 `RingSessionManager.swift`。以下是类的定义以及需要实现的关键属性：

```swift
@Observable
class RingSessionManager: NSObject {
    // 追踪连接状态
    var peripheralConnected = false
    var pickerDismissed = true

    // 存储当前连接的戒指
    var currentRing: ASAccessory?
    private var session = ASAccessorySession()

    // 核心蓝牙对象
    private var manager: CBCentralManager?
    private var peripheral: CBPeripheral?
}
```

#### 第二步：发现戒指

戒指通过特定的蓝牙服务 UUID 进行广播。为了找到它，我们需要创建一个 `ASDiscoveryDescriptor` 对象，指定其蓝牙服务的 UUID。以下代码完成了这一功能：

```swift
private static let ring: ASPickerDisplayItem = {
    let descriptor = ASDiscoveryDescriptor()
    descriptor.bluetoothServiceUUID = CBUUID(string: "6E40FFF0-B5A3-F393-E0A9-E50E24DCCA9E")
    return ASPickerDisplayItem(
        name: "COLMI R02 Ring",
        productImage: UIImage(named: "colmi")!,
        descriptor: descriptor
    )
}()
```

确保将戒指图片添加到项目资源目录中，或者用合适的占位符替换 `UIImage(named: "colmi")!`。

#### 第三步：显示戒指选择器

为了让用户选择戒指，我们调用系统内置的设备选择器界面：

```swift
func presentPicker() {
    session.showPicker(for: [Self.ring]) { error in
        if let error {
            print("Failed to show picker: \(error.localizedDescription)")
        }
    }
}
```

#### 第四步：处理戒指选择

当用户从选择器中选定设备后，应用需要处理连接和管理逻辑。以下代码实现了事件处理：

```swift
private func handleSessionEvent(event: ASAccessoryEvent) {
    switch event.eventType {
    case .accessoryAdded:
        guard let ring = event.accessory else { return }
        saveRing(ring: ring)
    case .activated:
        // 重新连接已配对戒指
        guard let ring = session.accessories.first else { return }
        saveRing(ring: ring)
    case .accessoryRemoved:
        currentRing = nil
        manager = nil
    }
}
```

#### 第五步：建立连接

完成选择戒指后，我们需要与其建立蓝牙连接：

```swift
func connect() {
    guard let manager, manager.state == .poweredOn, let peripheral else { return }
    let options: [String: Any] = [
        CBConnectPeripheralOptionNotifyOnConnectionKey: true,
        CBConnectPeripheralOptionNotifyOnDisconnectionKey: true,
        CBConnectPeripheralOptionStartDelayKey: 1
    ]
    manager.connect(peripheral, options: options)
}
```

#### 第六步：理解委托方法

在 `RingSessionManager` 中，我们实现了两个关键的委托协议，用于管理蓝牙通信过程。

**中央管理器委托（CBCentralManagerDelegate）**  
此委托主要处理蓝牙连接的整体状态。

```swift
func centralManagerDidUpdateState(_ central: CBCentralManager) {
    print("Central manager state: \(central.state)")
    switch central.state {
    case .poweredOn:
        if let peripheralUUID = currentRing?.bluetoothIdentifier {
            if let knownPeripheral = central.retrievePeripherals(withIdentifiers: [peripheralUUID]).first {
                print("Found previously connected peripheral")
                peripheral = knownPeripheral
                peripheral?.delegate = self
                connect()
            } else {
                print("Known peripheral not found, starting scan")
            }
        }
    default:
        peripheral = nil
    }
}
```

当蓝牙开启时，程序会检查是否有已连接的戒指，并尝试重新连接。  
成功连接后：

```swift
func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
    print("DEBUG: Connected to peripheral: \(peripheral)")
    peripheral.delegate = self
    print("DEBUG: Discovering services...")
    peripheral.discoverServices([CBUUID(string: Self.ringServiceUUID)])
    peripheralConnected = true
}
```

断开连接时：

```swift
func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: (any Error)?) {
    print("Disconnected from peripheral: \(peripheral)")
    peripheralConnected = false
    characteristicsDiscovered = false
}
```

**外设委托（CBPeripheralDelegate）**  

此委托主要处理与戒指的具体通信。  
首先发现戒指的服务：

```swift
func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: (any Error)?) {
    print("DEBUG: Services discovery callback, error: \(String(describing: error))")
    guard error == nil, let services = peripheral.services else {
        print("DEBUG: No services found or error occurred")
        return
    }
    print("DEBUG: Found \(services.count) services")
    for service in services {
        if service.uuid == CBUUID(string: Self.ringServiceUUID) {
            print("DEBUG: Found ring service, discovering characteristics...")
            peripheral.discoverCharacteristics([
                CBUUID(string: Self.uartRxCharacteristicUUID),
                CBUUID(string: Self.uartTxCharacteristicUUID)
            ], for: service)
        }
    }
}
```

发现特征后：

```swift
func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
    print("DEBUG: Characteristics discovery callback, error: \(String(describing: error))")
    guard error == nil, let characteristics = service.characteristics else {
        print("DEBUG: No characteristics found or error occurred")
        return
    }
    print("DEBUG: Found \(characteristics.count) characteristics")
    for characteristic in characteristics {
        switch characteristic.uuid {
        case CBUUID(string: Self.uartRxCharacteristicUUID):
            print("DEBUG: Found UART RX characteristic")
            self.uartRxCharacteristic = characteristic
        case CBUUID(string: Self.uartTxCharacteristicUUID):
            print("DEBUG: Found UART TX characteristic")
            self.uartTxCharacteristic = characteristic
            peripheral.setNotifyValue(true, for: characteristic)
        default:
            print("DEBUG: Found other characteristic: \(characteristic.uuid)")
        }
    }
    characteristicsDiscovered = true
}
```

接收数据时：

```swift
func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
    if characteristic.uuid == CBUUID(string: Self.uartTxCharacteristicUUID) {
        if let value = characteristic.value {
            print("Received value: \(value)")
        }
    }
}
```

发送命令后：

```swift
func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
    if let error = error {
        print("Write to characteristic failed: \(error.localizedDescription)")
    } else {
        print("Write to characteristic successful")
    }
}
```


##### 完整代码

完整的 `RingSessionManager` 类代码如下：

```swift
import Foundation
import AccessorySetupKit
import CoreBluetooth
import SwiftUI

@Observable
class RingSessionManager: NSObject {
    var peripheralConnected = false
    var pickerDismissed = true
    
    var currentRing: ASAccessory?
    private var session = ASAccessorySession()
    private var manager: CBCentralManager?
    private var peripheral: CBPeripheral?
    
    private var uartRxCharacteristic: CBCharacteristic?
    private var uartTxCharacteristic: CBCharacteristic?
    
    private static let ringServiceUUID = "6E40FFF0-B5A3-F393-E0A9-E50E24DCCA9E"
    private static let uartRxCharacteristicUUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
    private static let uartTxCharacteristicUUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
    
    private static let deviceInfoServiceUUID = "0000180A-0000-1000-8000-00805F9B34FB"
    private static let deviceHardwareUUID = "00002A27-0000-1000-8000-00805F9B34FB"
    private static let deviceFirmwareUUID = "00002A26-0000-1000-8000-00805F9B34FB"
    
    private static let ring: ASPickerDisplayItem = {
        let descriptor = ASDiscoveryDescriptor()
        descriptor.bluetoothServiceUUID = CBUUID(string: ringServiceUUID)
        
        return ASPickerDisplayItem(
            name: "COLMI R02 Ring",
            productImage: UIImage(named: "colmi")!,
            descriptor: descriptor
        )
    }()
    
    private var characteristicsDiscovered = false
    
    override init() {
        super.init()
        self.session.activate(on: DispatchQueue.main, eventHandler: handleSessionEvent(event:))
    }
    
    // MARK: - RingSessionManager actions
    func presentPicker() {
        session.showPicker(for: [Self.ring]) { error in
            if let error {
                print("Failed to show picker due to: \(error.localizedDescription)")
            }
        }
    }
    
    func removeRing() {
        guard let currentRing else { return }
        
        if peripheralConnected {
            disconnect()
        }
        
        session.removeAccessory(currentRing) { _ in
            self.currentRing = nil
            self.manager = nil
        }
    }
    
    func connect() {
        guard
            let manager, manager.state == .poweredOn,
            let peripheral
        else {
            return
        }
        let options: [String: Any] = [
            CBConnectPeripheralOptionNotifyOnConnectionKey: true,
            CBConnectPeripheralOptionNotifyOnDisconnectionKey: true,
            CBConnectPeripheralOptionStartDelayKey: 1
        ]
        manager.connect(peripheral, options: options)
    }
    
    func disconnect() {
        guard let peripheral, let manager else { return }
        manager.cancelPeripheralConnection(peripheral)
    }
    
    // MARK: - ASAccessorySession functions
    private func saveRing(ring: ASAccessory) {
        currentRing = ring
        
        if manager == nil {
            manager = CBCentralManager(delegate: self, queue: nil)
        }
    }
    
    private func handleSessionEvent(event: ASAccessoryEvent) {
        switch event.eventType {
        case .accessoryAdded, .accessoryChanged:
            guard let ring = event.accessory else { return }
            saveRing(ring: ring)
        case .activated:
            guard let ring = session.accessories.first else { return }
            saveRing(ring: ring)
        case .accessoryRemoved:
            self.currentRing = nil
            self.manager = nil
        case .pickerDidPresent:
            pickerDismissed = false
        case .pickerDidDismiss:
            pickerDismissed = true
        default:
            print("Received event type \(event.eventType)")
        }
    }
}

// MARK: - CBCentralManagerDelegate
extension RingSessionManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        print("Central manager state: \(central.state)")
        switch central.state {
        case .poweredOn:
            if let peripheralUUID = currentRing?.bluetoothIdentifier {
                if let knownPeripheral = central.retrievePeripherals(withIdentifiers: [peripheralUUID]).first {
                    print("Found previously connected peripheral")
                    peripheral = knownPeripheral
                    peripheral?.delegate = self
                    connect()
                } else {
                    print("Known peripheral not found, starting scan")
                }
            }
        default:
            peripheral = nil
        }
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("DEBUG: Connected to peripheral: \(peripheral)")
        peripheral.delegate = self
        print("DEBUG: Discovering services...")
        peripheral.discoverServices([CBUUID(string: Self.ringServiceUUID)])
        
        peripheralConnected = true
    }
    
    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: (any Error)?) {
        print("Disconnected from peripheral: \(peripheral)")
        peripheralConnected = false
        characteristicsDiscovered = false
    }
    
    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: (any Error)?) {
        print("Failed to connect to peripheral: \(peripheral), error: \(error.debugDescription)")
    }
}

// MARK: - CBPeripheralDelegate
extension RingSessionManager: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: (any Error)?) {
        print("DEBUG: Services discovery callback, error: \(String(describing: error))")
        guard error == nil, let services = peripheral.services else {
            print("DEBUG: No services found or error occurred")
            return
        }
        
        print("DEBUG: Found \(services.count) services")
        for service in services {
            if service.uuid == CBUUID(string: Self.ringServiceUUID) {
                print("DEBUG: Found ring service, discovering characteristics...")
                peripheral.discoverCharacteristics([
                    CBUUID(string: Self.uartRxCharacteristicUUID),
                    CBUUID(string: Self.uartTxCharacteristicUUID)
                ], for: service)
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        print("DEBUG: Characteristics discovery callback, error: \(String(describing: error))")
        guard error == nil, let characteristics = service.characteristics else {
            print("DEBUG: No characteristics found or error occurred")
            return
        }
        
        print("DEBUG: Found \(characteristics.count) characteristics")
        for characteristic in characteristics {
            switch characteristic.uuid {
            case CBUUID(string: Self.uartRxCharacteristicUUID):
                print("DEBUG: Found UART RX characteristic")
                self.uartRxCharacteristic = characteristic
            case CBUUID(string: Self.uartTxCharacteristicUUID):
                print("DEBUG: Found UART TX characteristic")
                self.uartTxCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
            default:
                print("DEBUG: Found other characteristic: \(characteristic.uuid)")
            }
        }
        characteristicsDiscovered = true
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        if characteristic.uuid == CBUUID(string: Self.uartTxCharacteristicUUID) {
            if let value = characteristic.value {
                print("Received value: \(value)")
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        if let error = error {
            print("Write to characteristic failed: \(error.localizedDescription)")
        } else {
            print("Write to characteristic successful")
        }
    }
}

```

### 最后一步：将其应用到我们的应用程序中

在 `ContentView.swift` 中粘贴以下代码，作为主界面的一部分：

```swift
import SwiftUI
import AccessorySetupKit

struct ContentView: View {
    @State var ringSessionManager = RingSessionManager()
    var body: some View {
        List {
            Section("MY DEVICE", content: {
                if ringSessionManager.pickerDismissed, let currentRing = ringSessionManager.currentRing {
                    makeRingView(ring: currentRing)
                } else {
                    Button {
                        ringSessionManager.presentPicker()
                    } label: {
                        Text("Add Ring")
                            .frame(maxWidth: .infinity)
                            .font(Font.headline.weight(.semibold))
                    }
                }
            })
        }.listStyle(.insetGrouped)
    }
    
    @ViewBuilder
    private func makeRingView(ring: ASAccessory) -> some View {
        HStack {
            Image("colmi")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(height: 70)
            VStack(alignment: .leading) {
                Text(ring.displayName)
                    .font(Font.headline.weight(.semibold))
            }
        }
    }
}

#Preview {
    ContentView()
}
```

如果一切配置正确，你现在可以构建并运行应用。当点击“Add Ring”按钮时，将弹出一个界面，显示附近的兼容设备（包括 COLMI R02 戒指）。选择设备后，应用即可完成连接。🎉

![](https://cdn-uploads.huggingface.co/production/uploads/66ba71a4447411b9c0e19d71/L7OYGOYxmfmrtbEagxyWu.png)


![连接演示](https://files.mdnice.com/user/38198/88ba809b-b14d-468a-8cf6-190e9db50b06.png)


在后续的文章中，我们将进一步探索如何与戒指交互，包括读取电池电量、获取传感器数据（如 PPG 和加速度计），并基于这些数据开发实时心率监测、活动追踪及睡眠检测功能。敬请期待！

> 英文原文: [https://hf.co/blog/cyrilzakka/halo-introduction](https://huggingface.co/blog/cyrilzakka/halo-introduction)
>
> 原文作者: Cyril, ML Researcher, Health AI Lead @ Hugging Face
>
> 译者: Lu Cheng, Hugging Face Fellow
