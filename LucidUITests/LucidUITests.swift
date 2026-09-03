//
//  LucidUITests.swift
//  LucidUITests
//
//  Lucid has no main window: it lives in the menu bar and the Dock menu, and
//  everything it draws goes into someone else's browser. There is very little
//  for a UI test to drive, so this checks the one thing that is worth checking
//  automatically - that the app launches, stays up, and does not claim a window.
//  Real verification is the unit tests, the offline bench, and the test lab.
//

import XCTest

final class LucidUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testLaunchesAsAnAccessoryAppAndStaysRunning() throws {
        let app = XCUIApplication()
        app.launch()
        XCTAssertEqual(app.state, .runningForeground)
        // A menu bar app should not be putting a window on screen at launch.
        XCTAssertEqual(app.windows.count, 0, "Lucid should not open a window when it starts")
    }
}
