import XCTest

final class LucidUITests: XCTestCase {
    func testLaunchDoesNotOpenControls() throws {
        let app = XCUIApplication()
        app.launch()
        defer { app.terminate() }
        XCTAssertTrue(app.state == .runningForeground || app.state == .runningBackground)
        XCTAssertEqual(app.windows.count, 0, "Lucid controls must open only from the menu bar icon")
        XCTAssertEqual(app.popovers.count, 0, "Launching Lucid must not open its dropdown")
    }
}
