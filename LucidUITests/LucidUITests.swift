import XCTest

final class LucidUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testManualLaunchUsesMenuBarDropdownAndStartupOption() throws {
        let app = XCUIApplication()
        app.launch()
        defer { app.terminate() }
        let startup = app.descendants(matching: .any).matching(identifier: "Launch at login").firstMatch
        XCTAssertTrue(startup.waitForExistence(timeout: 10), "Opening Lucid should reveal its menu bar dropdown")
        XCTAssertFalse(app.windows["Lucid"].exists, "Lucid must not open a separate controls window")
        XCTAssertTrue(startup.isEnabled)
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Lucid startup controls"
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
