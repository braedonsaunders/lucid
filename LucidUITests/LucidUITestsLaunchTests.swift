//
//  LucidUITestsLaunchTests.swift
//  LucidUITests
//
//  Captures a launch screenshot so a regression in start-up is visible in the
//  test report rather than only in a failure message.
//

import XCTest

final class LucidUITestsLaunchTests: XCTestCase {
    override class var runsForEachTargetApplicationUIConfiguration: Bool { false }

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testLaunchScreen() throws {
        let app = XCUIApplication()
        app.launch()

        let screenshot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        screenshot.name = "Launch"
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
