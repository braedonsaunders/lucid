import XCTest

final class LucidUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testManualLaunchRevealsControlsAndStartupOption() throws {
        let app = XCUIApplication()
        app.launch()
        defer { app.terminate() }
        let window = app.windows["Lucid"]
        XCTAssertTrue(window.waitForExistence(timeout: 10), "Opening Lucid should reveal its controls")
        let startup = window.descendants(matching: .any).matching(identifier: "Launch at login").firstMatch
        XCTAssertTrue(startup.exists, "Startup must be configurable in the visible controls")
        XCTAssertTrue(startup.isEnabled)
        let screenshot = XCTAttachment(screenshot: window.screenshot())
        screenshot.name = "Lucid startup controls"
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
