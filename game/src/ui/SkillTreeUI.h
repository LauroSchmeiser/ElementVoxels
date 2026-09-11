#pragma once

#include <imgui.h>
#include <array>
#include <string>
#include <algorithm>
#include <cmath>
#include <functional>

class SkillTreeUI
{
public:
    enum class Element
    {
        Fire,
        Water,
        Lightning,
        Crystal,
        Stone,
        Blood
    };

    struct Skill
    {
        const char* name;
        const char* description;
        int parent;
        int maxLevel;
        int level;
    };

    struct Page
    {
        const char* name;
        Element element;
        int level = 1;
        float xp = 0.0f;
        float xpRequired = 100.0f;
        int skillPoints = 1;
        std::array<Skill, 10> skills;
    };

    SkillTreeUI()
    {
        const char* names[6] =
                {
                        "Pyromancy",
                        "Aquamancy",
                        "Dynomancy",
                        "Telepathy",
                        "Geomancy",
                        "Hemomancy"
                };

        const Element elements[6] =
                {
                        Element::Fire,
                        Element::Water,
                        Element::Lightning,
                        Element::Crystal,
                        Element::Stone,
                        Element::Blood
                };

        for (int p = 0; p < 6; ++p) { pages[p].name = names[p]; pages[p].element = elements[p];}

        // Fire
        pages[0].skills =
                {{
                         { "Fire Resistance", "Take much less fire damage.", -1, 1, 0 },
                         { "Inner Flame", "You slowly heal, but your flesh is burned off instantly.", 0, 1, 0 },
                         { "Pyromancy", "Allows you to use fire for spells.", 0, 1, 0 },
                         { "Hellish Cyclone", "Hold to remove solid voxels and form a growing cyclone overhead. Release to strike enemies below with a fire laser.", 0, 1, 0 },
                         { "Inner Sun", "Fire spells can be cast without requiring material.", 1, 1, 0 },
                         { "Ignition Core", "Become immune to fire damage and slowly deal fire damage to enemies around you.", 1, 1, 0 },
                         { "Path of Destruction", "Convert materials you step on to fire and convert projectiles that touch you into fire.", 2, 1, 0 },
                         { "Pyroclasm", "Strengthen the destructive power of your fire spells.", 2, 1, 0 },
                         { "Inferno", "Greatly increase the destructive power and reach of your fire magic.", 3, 1, 0 },
                         { "Living Flame", "Become one with fire, greatly enhancing your fire abilities.", 3, 1, 0 }
                 }};

        // Water
        pages[1].skills =
                {{
                         { "Aqua Affinity", "Fluids can be used for spells. Move faster while in fluid.", -1, 1, 0 },
                         { "Conversion", "Turn solids within a radius into fluid.", 0, 1, 0 },
                         { "Thick Fluids", "Your fluid spells become denser, block solids, damage enemies, and slow them.", 0, 1, 0 },
                         { "Healing Water", "Fluids slowly heal you.", 0, 1, 0 },
                         { "Expansion", "Allows you to spread the fluid you are in to a radius.", 1, 1, 0 },
                         { "Tidal Force", "Greatly increase the scale and reach of your fluid spells.", 1, 1, 0 },
                         { "Stream", "Shoot a beam of compressed fluid that deals damage over time, knocks back enemies and bodies, damages the world, and removes voxels over time.", 2, 5, 0 },
                         { "Ocean's Wrath", "Master the destructive potential of water and fluid.", 2, 1, 0 },
                         { "Fluid Mastery", "All fluids now deal damage over time to enemies within them.", 3, 1, 0 },
                         { "Deep Current", "Enemies are pulled back into fluids when they enter them once.", 3, 1, 0 },
                 }};

        // Lightning
        pages[2].skills =
                {{
                         { "Enhanced Speed", "Allows you to increase your velocity for a short while.", -1, 1, 0 },
                         { "Electrical Current", "While Enhanced Speed is active, deal damage to enemies on contact.", 0, 1, 0 },
                         { "Reversed Momentum", "Reflecting a body with a punch adds additional speed to the body.", 0, 1, 0 },
                         { "Reaching Velocity", "Your spell range and cast speed increase with velocity.", 0, 1, 0 },
                         { "Inverted Velocity", "Every body in an area around you has its velocity inverted.", 1, 1, 0 },
                         { "Aerodynamic", "Accelerate much faster. Increase coyote time, cast speed, and melee damage with speed.", 1, 1, 0 },
                         { "Superspeed", "Slow down time for everyone except yourself for a short duration.", 2, 1, 0 },
                         { "Lightning Reflexes", "Further improve your movement and reaction speed.", 2, 1, 0 },
                         { "Velocity Mastery", "Push your movement speed and spellcasting velocity beyond normal limits.", 3, 1, 0 },
                         { "Absolute Speed", "Reach incredible speeds and gain mastery over velocity itself.", 3, 1, 0 }
                 }};

        // Telepathy / Crystal
        pages[3].skills =
                {{
                         { "Pull", "Allows you to pull a body towards you or pull yourself towards the world.", -1, 1, 0 },
                         { "Crystal Attunement", "Increase the effects and efficiency of crystal casts. Crystal spells are harder to destroy.", 0, 1, 0 },
                         { "Meditation", "While emerged in liquid crystal, take less damage and slowly heal.", 0, 1, 0 },
                         { "Spiritual Casting", "Gain access to spirit energy that can be used instead of material. Energy recharges over time and faster while meditating.", 0, 1, 0 },
                         { "Focused Chant", "Charge your spells to increase their size and speed while taking in material over time.", 1, 1, 0 },
                         { "Warding Crystals", "Your Crystals liquidize on impact and will solidify to ward of harm.", 1, 1, 0 },
                         { "Astral Projection", "Transmit your body to a different crystal surface or liquid crystal.", 2, 1, 0 },
                         { "Sixth Sense", "Gain better perception when emerged within liquid crystal or standing on crystal.", 2, 1, 0 },
                         { "Deep Reserves", "Greatly increase your Energy pool. Energy can also be replenished by casting crystal spells.", 3, 1, 0 },
                         { "Crystal Transcendence", "Master crystal and spirit energy, greatly enhancing your abilities.", 3, 1, 0 }
                 }};

        // Geomancy / Stone
        pages[4].skills =
                {{
                         { "Architecture", "Unlocks the forms Dome, Bridge, and Spear.", -1, 1, 0 },
                         { "Creation", "Your solid spells require much less material and smaller spells can be cast without any.", 0, 1, 0 },
                         { "Compressed Casts", "Your spells become much denser and faster.", 0, 1, 0 },
                         { "Stone Fist", "Your punch deals more damage when material is nearby.", 0, 1, 0 },
                         { "Grounded", "While standing on a solid surface, take much less damage.", 1, 1, 0 },
                         { "Compress", "Allows you to convert fluids into solids.", 1, 1, 0 },
                         { "Shrapnel", "Pull material towards you in a radius, creating many small bodies that damage enemies they hit.", 2, 1, 0 },
                         { "Stone Mastery", "Increase the strength and efficiency of your solid spells.", 2, 5, 0 },
                         { "Earth Fortress", "Become extremely resilient while surrounded by solid material.", 3, 1, 0 },
                         { "Geomantic Mastery", "Achieve mastery over stone, solids, and the world around you.", 3, 1, 0 }
                 }};

        // Hemomancy / Blood
        pages[5].skills =
                {{
                         { "Fiendish Resistance", "Flesh deals less damage to you. Heal while swimming in blood.", -1, 1, 0 },
                         { "Summon", "Summon a friendly unit to assist you in battle.", 0, 1, 0 },
                         { "Blood Aura", "Turn nearby flesh into blood.", 0, 1, 0 },
                         { "Blood Affinity", "Allows you to cast spells using blood and flesh.", 0, 1, 0 },
                         { "Sadism", "Killing enemies heals you.", 1, 1, 0 },
                         { "Blood Curse", "Touching blood curses enemies for a short duration, causing them to assist you in battle.", 1, 1, 0 },
                         { "Wrathful Spirit", "Dying while emerged in blood allows you to return with 50% health while absorbing nearby blood.", 2, 1, 0 },
                         { "Blood Mastery", "Increase the strength and efficiency of your blood abilities.", 2, 1, 0 },
                         { "Crimson Rebirth", "Gain greater control over blood and the boundary between life and death.", 3, 1, 0 },
                         { "Hemomantic Ascension", "Achieve mastery over blood, flesh, and life itself.", 3, 1, 0 }
                 }};
    }

    void Reset()
    {
        for (Page& page : pages)
        {
            page.level = 1;
            page.xp = 0.0f;
            page.xpRequired = 100.0f;
            page.skillPoints = 1;

            for (Skill& skill : page.skills)
                skill.level = 0;
        }

        currentPage = 0;
        hoveredSkill = -1;
        selectedSkill = -1;
    }

    void Draw() {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        const ImVec2 viewportPos = viewport->WorkPos;
        const ImVec2 viewportSize = viewport->WorkSize;
        const ImVec2 windowSize( viewportSize.x * 0.75f,viewportSize.y * 0.75f );
        const ImVec2 windowPos( viewportPos.x + (viewportSize.x - windowSize.x) * 0.5f,
                                viewportPos.y + (viewportSize.y - windowSize.y) * 0.5f );

        ImGui::SetNextWindowPos( windowPos, ImGuiCond_Always );
        ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize
                | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;

        ImGui::Begin( "Skill Trees", nullptr, flags );
                DrawHeader();
                ImGui::Spacing();

                DrawTabs();
                ImGui::Spacing();

                DrawXPBar();
                ImGui::Spacing();

                ImGui::Separator();
                ImGui::Spacing();

                DrawTree();
                ImGui::Spacing();

                ImGui::Separator();
                ImGui::Spacing();

                DrawSelectedSkill();

        const ImVec2 btnSize(520.0f * 0.75f, 42.0f);

        if (ImGui::Button("give XP", btnSize))
        {
            AddXP(1000000000000.0f);

        }

        ImGui::End();
    }

    Page& GetPage(int index)
    {
        return pages[std::clamp(index, 0, 5)];
    }

    void SetPage(int index)
    {
        currentPage = std::clamp(index, 0, 5);
    }


    Page& GetCurrentPage()
    {
        return pages[currentPage];
    }

    void AddXP(float amount)
    {
        Page& page = pages[currentPage];

        page.xp += amount;

        while (page.xp >= page.xpRequired)
        {
            page.xp -= page.xpRequired;
            ++page.level;
            ++page.skillPoints;
            page.xpRequired *= 1.15f;
        }
    }

    using MainMenuCallback = std::function<void()>;
    void SetMainMenuCallback(MainMenuCallback callback) { mainMenuCallback = std::move(callback); }

    void GiveLevel(int page)
    {
        if (page < 0 || page >= 6)
            return;

        Page& skillPage = pages[page];

        ++skillPage.level;
        ++skillPage.skillPoints;
        skillPage.xp = 0.0f;
    }

private:
    MainMenuCallback mainMenuCallback;
    std::array<Page, 6> pages;
    int currentPage = 0;
    int hoveredSkill = -1;
    int selectedSkill = -1;

    const ImVec2 nodePositions[10] =
            {
                    ImVec2(0.50f, 0.15f),

                    ImVec2(0.28f, 0.42f),
                    ImVec2(0.50f, 0.42f),
                    ImVec2(0.72f, 0.42f),

                    ImVec2(0.20f, 0.72f),
                    ImVec2(0.36f, 0.72f),

                    ImVec2(0.44f, 0.72f),
                    ImVec2(0.56f, 0.72f),

                    ImVec2(0.64f, 0.72f),
                    ImVec2(0.80f, 0.72f)
            };

    void DrawTabs()
    {
        for (int i = 0; i < 6; ++i)
        {
            if (i > 0)
                ImGui::SameLine();

            bool selected = currentPage == i;

            if (selected)
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.20f, 0.25f, 1.0f));

            if (ImGui::Button(pages[i].name, ImVec2(90.0f, 32.0f)))
            {
                currentPage = i;
                hoveredSkill = -1;
                selectedSkill = -1;
            }

            if (selected)
                ImGui::PopStyleColor();
        }
    }

    void DrawXPBar()
    {
        Page& page = pages[currentPage];

        ImGui::Text("Level %d", page.level);
        ImGui::SameLine();

        float percent = page.xpRequired > 0.0f
                        ? page.xp / page.xpRequired
                        : 0.0f;

        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%.0f / %.0f XP", page.xp, page.xpRequired);

        ImGui::ProgressBar(percent, ImVec2(-1.0f, 22.0f), buffer);

        ImGui::Text("Skill Points: %d", page.skillPoints);
    }

    bool IsVisible(int index) const
    {
        const Page& page = pages[currentPage];
        const Skill& skill = page.skills[index];

        if (skill.parent == -1)
            return true;

        return page.skills[skill.parent].level > 0;
    }

    bool CanUpgrade(int index) const
    {
        const Page& page = pages[currentPage];
        const Skill& skill = page.skills[index];

        if (!IsVisible(index))
            return false;

        if (skill.level >= skill.maxLevel)
            return false;

        if (page.skillPoints <= 0)
            return false;

        return true;
    }

    ImVec4 GetElementColor(Element element) const
    {
        switch (element)
        {
            case Element::Fire:
                return ImVec4(1.0f, 0.35f, 0.08f, 1.0f);

            case Element::Water:
                return ImVec4(0.15f, 0.55f, 1.0f, 1.0f);

            case Element::Lightning:
                return ImVec4(1.0f, 0.85f, 0.15f, 1.0f);

            case Element::Crystal:
                return ImVec4(0.65f, 0.35f, 1.0f, 1.0f);

            case Element::Stone:
                return ImVec4(0.55f, 0.55f, 0.58f, 1.0f);

            case Element::Blood:
                return ImVec4(0.85f, 0.08f, 0.12f, 1.0f);
        }

        return ImVec4(1, 1, 1, 1);
    }

    ImU32 ToU32(ImVec4 color, float alpha = 1.0f) const
    {
        color.w *= alpha;
        return ImGui::ColorConvertFloat4ToU32(color);
    }

    void DrawTree()
    {
        Page& page = pages[currentPage];

        ImVec2 available = ImGui::GetContentRegionAvail();

        float treeHeight = std::max(420.0f, available.y - 120.0f);
        float treeWidth = available.x;

        ImVec2 origin = ImGui::GetCursorScreenPos();

        ImGui::InvisibleButton(
                "##skill_tree_canvas",
                ImVec2(treeWidth, treeHeight)
        );

        ImDrawList* draw = ImGui::GetWindowDrawList();

        hoveredSkill = -1;

        ImVec2 positions[10];

        for (int i = 0; i < 10; ++i)
        {
            positions[i] =
                    {
                            origin.x + nodePositions[i].x * treeWidth,
                            origin.y + nodePositions[i].y * treeHeight
                    };
        }

        for (int i = 0; i < 10; ++i)
        {
            int parent = page.skills[i].parent;

            if (parent < 0)
                continue;

            ImVec2 a = positions[parent];
            ImVec2 b = positions[i];

            bool active =
                    page.skills[parent].level > 0 &&
                    IsVisible(i);

            ImVec4 color =
                    active
                    ? GetElementColor(page.element)
                    : ImVec4(0.22f, 0.24f, 0.27f, 1.0f);

            draw->AddLine(
                    a,
                    b,
                    ToU32(color, active ? 0.75f : 0.35f),
                    3.0f
            );
        }

        for (int i = 0; i < 10; ++i)
            DrawNode(draw, positions[i], i);
    }

    void DrawNode(ImDrawList* draw, ImVec2 pos, int index)
    {
        Page& page = pages[currentPage];
        Skill& skill = page.skills[index];

        bool visible = IsVisible(index);

        float radius = 27.0f;

        ImVec2 min(pos.x - radius, pos.y - radius);
        ImVec2 max(pos.x + radius, pos.y + radius);

        bool hovered = ImGui::IsMouseHoveringRect(min, max);

        if (hovered && visible)
            hoveredSkill = index;

        bool unlocked = skill.level > 0;
        bool available = CanUpgrade(index);
        bool selected = selectedSkill == index;

        ImVec4 elementColor = GetElementColor(page.element);

        if (hovered && visible)
        {
            draw->AddCircleFilled(
                    pos,
                    radius + 12.0f,
                    ToU32(elementColor, 0.10f)
            );

            draw->AddCircleFilled(
                    pos,
                    radius + 7.0f,
                    ToU32(elementColor, 0.16f)
            );
        }

        if (available)
        {
            float pulse =
                    1.5f +
                    std::sin(static_cast<float>(ImGui::GetTime()) * 3.0f) * 1.5f;

            draw->AddCircle(
                    pos,
                    radius + 5.0f + pulse,
                    ToU32(elementColor, 0.35f),
                    32,
                    2.0f
            );
        }

        if (!visible)
        {
            draw->AddCircleFilled(
                    pos,
                    radius,
                    IM_COL32(38, 41, 45, 255)
            );

            draw->AddCircle(
                    pos,
                    radius,
                    IM_COL32(70, 74, 80, 255),
                    32,
                    2.0f
            );

            const char* question = "?";
            ImVec2 size = ImGui::CalcTextSize(question);

            draw->AddText(
                    ImVec2(pos.x - size.x * 0.5f, pos.y - size.y * 0.5f),
                    IM_COL32(130, 135, 140, 255),
                    question
            );

            return;
        }

        ImVec4 fillColor;

        if (unlocked)
            fillColor = elementColor;
        else
            fillColor = ImVec4(0.14f, 0.16f, 0.19f, 1.0f);

        draw->AddCircleFilled(
                pos,
                radius,
                ToU32(fillColor)
        );

        ImVec4 borderColor =
                selected
                ? ImVec4(1.0f, 0.85f, 0.3f, 1.0f)
                : elementColor;

        draw->AddCircle(
                pos,
                radius,
                ToU32(borderColor),
                32,
                selected ? 4.0f : 2.5f
        );

        DrawIcon(
                draw,
                pos,
                page.element,
                unlocked ? 1.0f : 0.45f
        );

        if (skill.level > 0)
        {
            char levelText[16];
            snprintf(
                    levelText,
                    sizeof(levelText),
                    "%d/%d",
                    skill.level,
                    skill.maxLevel
            );

            ImVec2 size = ImGui::CalcTextSize(levelText);

            draw->AddRectFilled(
                    ImVec2(
                            pos.x - size.x * 0.5f - 5.0f,
                            pos.y + radius - 2.0f
                    ),
                    ImVec2(
                            pos.x + size.x * 0.5f + 5.0f,
                            pos.y + radius + 15.0f
                    ),
                    IM_COL32(20, 22, 25, 230),
                    5.0f
            );

            draw->AddText(
                    ImVec2(
                            pos.x - size.x * 0.5f,
                            pos.y + radius
                    ),
                    IM_COL32(230, 230, 230, 255),
                    levelText
            );
        }

        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        {
            selectedSkill = index;

            if (CanUpgrade(index))
            {
                ++skill.level;
                --page.skillPoints;
            }
        }

        if (hovered)
            DrawTooltip(index);
    }

    void DrawTooltip(int index)
    {
        Page& page = pages[currentPage];
        Skill& skill = page.skills[index];

        ImGui::BeginTooltip();

        ImGui::TextColored(
                GetElementColor(page.element),
                "%s",
                skill.name
        );

        ImGui::Separator();

        ImGui::Text(
                "Level %d / %d",
                skill.level,
                skill.maxLevel
        );

        ImGui::Spacing();

        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 25.0f);

        ImGui::Text(
                "%s",
                skill.description
        );

        ImGui::PopTextWrapPos();

        ImGui::Spacing();

        if (skill.level >= skill.maxLevel)
        {
            ImGui::TextDisabled("Max level");
        }
        else if (CanUpgrade(index))
        {
            ImGui::TextColored(
                    ImVec4(0.4f, 1.0f, 0.45f, 1.0f),
                    "Click to upgrade"
            );
        }
        else
        {
            ImGui::TextDisabled("Requires a skill point");
        }

        ImGui::EndTooltip();
    }

    void DrawSelectedSkill()
    {
        if (selectedSkill < 0)
            return;

        Page& page = pages[currentPage];
        Skill& skill = page.skills[selectedSkill];

        ImGui::TextColored(
                GetElementColor(page.element),
                "%s",
                skill.name
        );

        ImGui::Text(
                "Level %d / %d",
                skill.level,
                skill.maxLevel
        );

        ImGui::TextWrapped(
                "%s",
                skill.description
        );

        if (CanUpgrade(selectedSkill))
        {
            ImGui::Spacing();

            if (ImGui::Button("Upgrade"))
            {
                ++skill.level;
                --page.skillPoints;
            }
        }
    }

    void DrawHeader() {
        if (ImGui::Button("< Back to Main Menu", ImVec2(170.0f, 32.0f)))
        { if (mainMenuCallback) mainMenuCallback(); }
        ImGui::SameLine();
        const char* title = pages[currentPage].name;
        float width = ImGui::GetContentRegionAvail().x;
        float textWidth = ImGui::CalcTextSize(title).x;
        ImGui::SetCursorPosX( ImGui::GetCursorPosX() + (width - textWidth) * 0.5f );
        ImGui::Text( "%s", title );
    }

    void DrawIcon(
            ImDrawList* draw,
            ImVec2 center,
            Element element,
            float alpha
    )
    {
        ImVec4 color = GetElementColor(element);
        ImU32 col = ToU32(color, alpha);

        switch (element)
        {
            case Element::Fire:
            {
                ImVec2 points[] =
                        {
                                { center.x,       center.y - 16.0f },
                                { center.x + 10.0f, center.y - 2.0f },
                                { center.x + 7.0f,  center.y + 13.0f },
                                { center.x,        center.y + 17.0f },
                                { center.x - 8.0f,  center.y + 11.0f },
                                { center.x - 10.0f, center.y },
                                { center.x - 3.0f,  center.y + 4.0f }
                        };

                draw->AddConvexPolyFilled(points, 7, col);
                break;
            }

            case Element::Water:
            {
                ImVec2 points[] =
                        {
                                { center.x,        center.y - 17.0f },
                                { center.x + 12.0f, center.y + 3.0f },
                                { center.x + 9.0f,  center.y + 11.0f },
                                { center.x,         center.y + 16.0f },
                                { center.x - 9.0f,  center.y + 11.0f },
                                { center.x - 12.0f, center.y + 3.0f }
                        };

                draw->AddConvexPolyFilled(points, 6, col);
                break;
            }

            case Element::Lightning:
            {
                ImVec2 points[] =
                        {
                                { center.x + 3.0f,  center.y - 18.0f },
                                { center.x - 12.0f, center.y + 1.0f },
                                { center.x - 2.0f,  center.y + 1.0f },
                                { center.x - 7.0f,  center.y + 18.0f },
                                { center.x + 13.0f, center.y - 5.0f },
                                { center.x + 3.0f,  center.y - 5.0f }
                        };

                draw->AddConvexPolyFilled(points, 6, col);
                break;
            }

            case Element::Crystal:
            {
                ImVec2 points[] =
                        {
                                { center.x,        center.y - 18.0f },
                                { center.x + 13.0f, center.y - 7.0f },
                                { center.x + 9.0f,  center.y + 12.0f },
                                { center.x,         center.y + 18.0f },
                                { center.x - 9.0f,  center.y + 12.0f },
                                { center.x - 13.0f, center.y - 7.0f }
                        };

                draw->AddConvexPolyFilled(points, 6, col);
                break;
            }

            case Element::Stone:
            {
                ImVec2 points[] =
                        {
                                { center.x - 13.0f, center.y - 9.0f },
                                { center.x - 4.0f,  center.y - 16.0f },
                                { center.x + 11.0f, center.y - 11.0f },
                                { center.x + 15.0f, center.y + 5.0f },
                                { center.x + 5.0f,  center.y + 15.0f },
                                { center.x - 11.0f, center.y + 12.0f }
                        };

                draw->AddConvexPolyFilled(points, 6, col);
                break;
            }

            case Element::Blood:
            {
                ImVec2 points[] =
                        {
                                { center.x,        center.y - 17.0f },
                                { center.x + 11.0f, center.y + 2.0f },
                                { center.x + 9.0f,  center.y + 11.0f },
                                { center.x,         center.y + 17.0f },
                                { center.x - 9.0f,  center.y + 11.0f },
                                { center.x - 11.0f, center.y + 2.0f }
                        };

                draw->AddConvexPolyFilled(points, 6, col);
                break;
            }
        }
    }
};

