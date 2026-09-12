import Foundation
import ServiceManagement
import Testing
@testable import Lucid

@MainActor
struct LoginItemControllerTests {
    @Test func externalRevocationIsReflectedWithoutChangingRegistration() {
        var status: SMAppService.Status = .enabled
        let item = LoginItemController(readStatus: { status },
                                       register: { Issue.record("Unexpected registration") },
                                       unregister: { Issue.record("Unexpected removal") })
        #expect(item.isEnabled)
        status = .requiresApproval
        item.refresh()
        #expect(!item.isEnabled)
        #expect(item.requiresApproval)
    }

    @Test func deniedPermissionOpensSettingsInsteadOfClaimingSuccess() {
        var opened = false
        let item = LoginItemController(readStatus: { .requiresApproval },
                                       register: { Issue.record("Must not re-register a denied item") },
                                       openSettings: { opened = true })
        item.setEnabled(true)
        #expect(opened)
        #expect(!item.isEnabled)
        #expect(item.requiresApproval)
    }

    @Test func failedRegistrationKeepsSwitchOffAndReportsError() {
        let item = LoginItemController(readStatus: { .notRegistered },
                                       register: { throw NSError(domain: "Test", code: 1) })
        item.setEnabled(true)
        #expect(!item.isEnabled)
        #expect(item.errorMessage != nil)
    }

    @Test func failedRemovalKeepsSwitchOnAndCanBeRetried() {
        var status: SMAppService.Status = .enabled
        var shouldFail = true
        let item = LoginItemController(readStatus: { status }, unregister: {
            if shouldFail { throw NSError(domain: "Test", code: 2) }
            status = .notRegistered
        })
        item.setEnabled(false)
        #expect(item.isEnabled)
        #expect(item.errorMessage != nil)
        shouldFail = false
        item.setEnabled(false)
        #expect(!item.isEnabled)
        #expect(item.errorMessage == nil)
    }

    @Test func registrationCanRequireApprovalWithoutThrowing() {
        var status: SMAppService.Status = .notRegistered
        let item = LoginItemController(readStatus: { status }, register: { status = .requiresApproval })
        item.setEnabled(true)
        #expect(!item.isEnabled)
        #expect(item.requiresApproval)
        #expect(item.errorMessage == nil)
    }

    @Test func alreadyEnabledIsNotRegisteredTwice() {
        let item = LoginItemController(readStatus: { .enabled },
                                       register: { Issue.record("Already registered") })
        item.setEnabled(true)
        #expect(item.isEnabled)
        #expect(item.errorMessage == nil)
    }
}
