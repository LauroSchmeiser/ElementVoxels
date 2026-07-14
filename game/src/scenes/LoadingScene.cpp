#include "LoadingScene.h"
#include "../Game.h"
#include "../SceneId.h"

#include <imgui.h>
#include <GLFW/glfw3.h>

namespace gl3 {

    void LoadingScene::onEnter(Game& game)
    {
        progress = 0.0f;
        t = 0.0;

        // show cursor (optional)
        //glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

       // game.setupSkybox();
        //game.bakeNebulaCubemap(512);
        // start preload pipeline
        game.beginGameplayPreload(/*newRun=*/true);
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    }

    void LoadingScene::onExit(Game& game)
    {
        //glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        std::cout<<"HelloWorld";
    }

    void LoadingScene::update(Game& game, float dt)
    {
        t += dt;

        // Do a bit of loading each frame
        progress = game.tickGameplayPreload();

        // When ready -> go to gameplay
       /* if (progress >= 1.0f) {
            game.requestSceneChange(SceneId::Gameplay);
            return;
        }*/
    }

    void LoadingScene::render(Game& game)
    {
        glViewport(0, 0, game.getWindowWidth(), game.getWindowHeight());
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_LEQUAL);
        game.renderSkybox();
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        game.imguiLayer.beginFrame();

        // ==== SPELLCRAFT MAIN WINDOW ====
        ImGuiIO& io = ImGui::GetIO();
        const ImVec2 display = io.DisplaySize;

        const float marginX = 24.0f;
        const float marginTop = 24.0f;
        const float bottomReserved = 120.0f; // keep space for loading bar row

        ImGui::SetNextWindowPos(ImVec2(marginX, marginTop), ImGuiCond_Always);
        ImGui::SetNextWindowSize(
                ImVec2(display.x - marginX * 2.0f, display.y - marginTop - bottomReserved),
                ImGuiCond_Always
        );
        ImGui::Begin("Spellcraft", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);

        ImGui::TextUnformatted("Create 3 Spell Presets");
        ImGui::Separator();

        // Top: 3 slots
        for (int i = 0; i < 3; ++i) {
            if (i > 0) ImGui::SameLine();
            std::string label = "Slot " + std::to_string(i + 1);
            if (ImGui::Selectable(label.c_str(), selectedSlot == i, 0, ImVec2(180, 60))) {
                selectedSlot = i;
            }

            const auto& sp = game.getSpellPreset(i);
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Name: %s", sp.name.c_str());
                ImGui::Text("Range: %.1f", sp.range);
                ImGui::Text("Cooldown: %.2fs", sp.cooldown);
                ImGui::Text("Cost: %.1f", sp.materialCost);
                ImGui::EndTooltip();
            }
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        auto& p = game.getSpellPreset(selectedSlot);

        // Categories as clickable labeled rectangles
        if (ImGui::Selectable("Material", selectedCategory == 0, 0, ImVec2(180, 36))) selectedCategory = 0;
        ImGui::SameLine();
        if (ImGui::Selectable("Form", selectedCategory == 1, 0, ImVec2(180, 36))) selectedCategory = 1;
        ImGui::SameLine();
        if (ImGui::Selectable("Size", selectedCategory == 2, 0, ImVec2(180, 36))) selectedCategory = 2;
        ImGui::SameLine();
        if (ImGui::Selectable("Range", selectedCategory == 3, 0, ImVec2(180, 36))) selectedCategory = 3;
        ImGui::SameLine();
        if (ImGui::Selectable("SpellType", selectedCategory == 4, 0, ImVec2(180, 36))) selectedCategory = 4;

        ImGui::Spacing();
        ImGui::BeginChild("CategoryEditor", ImVec2(0, 260), true);

        if (selectedCategory == 0) {
            ImGui::Text("Pick Material");
            if (ImGui::Button("Rock"))  p.material = Game::CraftMaterial::Rock;
            ImGui::SameLine();
            if (ImGui::Button("Flesh")) p.material = Game::CraftMaterial::Flesh;
            ImGui::SameLine();
            if (ImGui::Button("Lava"))  p.material = Game::CraftMaterial::Lava;

            if (ImGui::IsItemHovered()) { /* tooltip per item; split buttons if needed */ }
        }
        else if (selectedCategory == 1) {
            int form = (int)p.form;
            const char* forms[] = {"Sphere", "Wall"};
            if (ImGui::Combo("Form", &form, forms, IM_ARRAYSIZE(forms)))
                p.form = (Game::CraftForm)form;
        }
        else if (selectedCategory == 2) {
            if (p.form == Game::CraftForm::Sphere) {
                ImGui::SliderFloat("Radius", &p.radius, 0.5f, 12.0f);
            } else {
                ImGui::SliderFloat("Width", &p.width, 0.5f, 20.0f);
                ImGui::SliderFloat("Height", &p.height, 0.5f, 20.0f);
                ImGui::SliderFloat("Thickness", &p.thickness, 0.2f, 8.0f);
            }
        }
        else if (selectedCategory == 3) {
            ImGui::SliderFloat("Range", &p.range, 5.0f, 150.0f);
        }
        else if (selectedCategory == 4) {
            int st = (int)p.spellType;
            const char* types[] = {"Construct (Static)", "Projectile (Dynamic)"};
            if (ImGui::Combo("Spell Type", &st, types, IM_ARRAYSIZE(types)))
                p.spellType = (Game::CraftSpellType)st;
        }

        game.recomputeSpellDerivedStats(p);

        ImGui::EndChild();

        // Always-visible stats bar at bottom of spellcraft panel
        ImGui::Separator();
        ImGui::Text("Stats  |  Cooldown: %.2fs   Material Cost: %.1f   Range: %.1f",
                    p.cooldown, p.materialCost, p.range);

        ImGui::End();

        // ==== LOADING BAR WINDOW BOTTOM ====
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y - 110), ImGuiCond_Always, ImVec2(0.5f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(720, 90), ImGuiCond_Always);

        ImGui::Begin("LoadingBottom", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

        ImGui::TextUnformatted(game.getGameplayPreloadStageName().c_str());

        // left: progress bar (dynamic width), right: button
        const float btnW = 180.0f;
        const bool ready = (progress >= 1.0f);
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float barW = ImGui::GetContentRegionAvail().x - (ready ? (btnW + spacing) : 0.0f);

        ImGui::ProgressBar(progress, ImVec2(barW, 24.0f));

        if (ready) {
            game.requestSceneChange(SceneId::Gameplay);

           /* ImGui::SameLine();
            if (ImGui::Button("Enter Game", ImVec2(btnW, 24.0f))) {
                game.requestSceneChange(SceneId::Gameplay);
            }*/
        }

        ImGui::End();

        game.imguiLayer.endFrame();
    }
} // namespace gl3